import os
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import config
import textwrap
import matplotlib.gridspec as gridspec
import numpy as np

from problem.TrussFeatures import TrussFeatures, intersect, calculate_angle, estimate_intersection_volume



# Calculate bit connection list
bit_connection_list = []

for x in range(config.num_vars):
    design = [0 for _ in range(config.num_vars)]
    design[x] = 1
    truss_features = TrussFeatures(design, config.sidenum, None)
    design_conn_array = truss_features.design_conn_array
    bit_connection_list.append(design_conn_array)

    # print('--> Bit:', x, 'Connections:', design_conn_array)





class VolFrac:

    def __init__(self, sidenum, bit_list):
        self.bit_list = bit_list
        self.sidenum = sidenum
        # self.num_members = TrussFeatures.get_member_count(self.sidenum)
        # self.num_repeatable_members = TrussFeatures.get_member_count_repeatable(self.sidenum)

        if self.sidenum == 3:
            self.n = 30
            self.feasibility_constraint_norm = 94
        elif self.sidenum == 4:
            self.n = 108
            self.feasibility_constraint_norm = 1304
        elif self.sidenum == 5:
            self.n = 280  # TODO: get the actual value
            self.feasibility_constraint_norm = 8942
        elif self.sidenum == 6:
            self.n = 600  # TODO: get the actual value
            self.feasibility_constraint_norm = 41397
        else:
            raise ValueError('Invalid sidenum:', self.sidenum)

        self.truss_features = TrussFeatures(bit_list, sidenum, None)
        self.design_conn_array = self.truss_features.design_conn_array

        self.node_positions = {}
        idx = 1
        for x in range(self.sidenum):
            for y in range(self.sidenum):
                self.node_positions[idx] = (x, y)
                idx += 1


    def visualize(self):
        self.truss_features.visualize_design()


    def get_bit_connections(self, bit_idx):
        return bit_connection_list[bit_idx]


    def get_bits_from_connection(self, connection):
        bits = []
        for idx, bit_conns in enumerate(bit_connection_list):
            for bit_conn in bit_conns:
                if connection[0] in bit_conn and connection[1] in bit_conn:
                    bits.append(idx)
        return bits

    def get_overlap_bits(self):

        curr_bits = []
        curr_conns = []
        overlap_bits = []
        for idx, bit in enumerate(self.bit_list):
            curr_bits.append(bit)
            if bit == 0:
                overlap_bits.append(0)
            else:
                curr_conns += self.get_bit_connections(idx)
                # Check if overlaps










        pass




    def evaluate(self, member_radii=50e-6, side_length=100e-5):

        # 1. calculate total volume of truss
        depth = member_radii * 2
        width = side_length * (self.sidenum - 1)
        height = side_length * (self.sidenum - 1)
        total_volume = depth * width * height

        # 2. calculate total volume of truss members
        member_volumes = []
        member_lengths = []
        member_positions = []
        for connection in self.design_conn_array:
            node1, node2 = connection
            pos1, pos2 = self.node_positions[node1], self.node_positions[node2]
            pos1 = [x * side_length for x in pos1]
            pos2 = [x * side_length for x in pos2]
            member_positions.append((pos1, pos2))
            length = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            member_lengths.append(length)
            member_volume = math.pi * member_radii**2 * length
            member_volumes.append(member_volume)

        # 3. Account for truss member overlaps at nodes
        node_volumes = []
        node_volume = (4 / 3) * math.pi * member_radii ** 3
        sub_vol = math.pi * member_radii ** 2 * member_radii
        num_nodes = self.sidenum ** 2
        for x in range(1, num_nodes+1):
            connected_members_indices = [i for i, connection in enumerate(self.design_conn_array) if x in connection]
            if len(connected_members_indices) > 0:
                node_volumes.append(node_volume)
            for idx in connected_members_indices:
                member_volumes[idx] -= sub_vol

        # 4. Account for non-node truss member overlaps
        intersect_vols = []
        num_intersections = 0
        bit_interactions = {}
        for member_a_idx in range(len(self.design_conn_array)):
            member_a_n1, member_a_n2 = self.design_conn_array[member_a_idx]
            a_pos_1, a_pos_2 = self.node_positions[member_a_n1], self.node_positions[member_a_n2]

            # Get bit idx of member a
            # print(self.design_conn_array[member_a_idx])
            bits_a = self.get_bits_from_connection(self.design_conn_array[member_a_idx])

            for member_b_idx in range(member_a_idx + 1, len(self.design_conn_array)):

                # Get bit idx of member b
                bits_b = self.get_bits_from_connection(self.design_conn_array[member_b_idx])

                member_b_n1, member_b_n2 = self.design_conn_array[member_b_idx]
                b_pos_1, b_pos_2 = self.node_positions[member_b_n1], self.node_positions[member_b_n2]
                intersects = intersect(a_pos_1, a_pos_2, b_pos_1, b_pos_2)
                if intersects is True:
                    # print('--> INTERSECTS:', self.design_conn_array[member_a_idx], self.design_conn_array[member_b_idx])
                    # print('-- MORE INFO:', a_pos_1, a_pos_2, b_pos_1, b_pos_2)
                    num_intersections += 1
                    angle = calculate_angle(a_pos_1, a_pos_2, b_pos_1, b_pos_2)
                    volume = estimate_intersection_volume(member_radii, angle)
                    intersect_vols.append(volume)
                    # print('Interaction: ', bits_a, bits_b)
                    for a_bit in bits_a:
                        for b_bit in bits_b:
                            if a_bit not in bit_interactions:
                                bit_interactions[a_bit] = []
                            if b_bit not in bit_interactions:
                                bit_interactions[b_bit] = []
                            if b_bit not in bit_interactions[a_bit]:
                                bit_interactions[a_bit].append(b_bit)
                            if a_bit not in bit_interactions[b_bit]:
                                bit_interactions[b_bit].append(a_bit)


        # 5. Account for parallel truss member overlaps
        y_axis_nodes = [[] for _ in range(self.sidenum)]
        x_axis_nodes = [[] for _ in range(self.sidenum)]
        for node in range(1, num_nodes + 1):
            x, y = node % self.sidenum, (node - 1) // self.sidenum
            y_axis_nodes[x].append(node)
            x_axis_nodes[y].append(node)
        y_axis_nodes = sorted(y_axis_nodes, key=lambda x: x[0])
        all_axis_nodes = x_axis_nodes + y_axis_nodes
        truss_members_vol_ignore = []
        axis_vols = []
        for axis_nodes in all_axis_nodes:
            axis_vol = 0
            axis_node_ranges = []
            for idx, connection in enumerate(self.design_conn_array):
                if connection[0] in axis_nodes and connection[1] in axis_nodes:
                    if idx not in truss_members_vol_ignore:
                        truss_members_vol_ignore.append(idx)
                    axis_node_idx_1 = axis_nodes.index(connection[0])
                    axis_node_idx_2 = axis_nodes.index(connection[1])
                    min_axis_node_idx = min(axis_node_idx_1, axis_node_idx_2)
                    max_axis_node_idx = max(axis_node_idx_1, axis_node_idx_2)
                    axis_node_range = [min_axis_node_idx, max_axis_node_idx]
                    if len(axis_node_ranges) == 0:
                        axis_node_ranges.append(axis_node_range)
                    else:
                        new_range = True
                        for axis_node_range in axis_node_ranges:
                            if axis_node_range[0] <= min_axis_node_idx <= axis_node_range[1] or axis_node_range[0] <= max_axis_node_idx <= axis_node_range[1]:
                                if new_range is False:
                                    num_intersections += 1
                                    bits_a = self.get_bits_from_connection(self.design_conn_array[idx])
                                    bits_b = self.get_bits_from_connection(self.design_conn_array[axis_node_range[0]])
                                    for a_bit in bits_a:
                                        for b_bit in bits_b:
                                            if a_bit not in bit_interactions:
                                                bit_interactions[a_bit] = []
                                            if b_bit not in bit_interactions:
                                                bit_interactions[b_bit] = []
                                            if b_bit not in bit_interactions[a_bit]:
                                                bit_interactions[a_bit].append(b_bit)
                                            if a_bit not in bit_interactions[b_bit]:
                                                bit_interactions[b_bit].append(a_bit)
                                new_range = False
                                if min_axis_node_idx < axis_node_range[0]:
                                    axis_node_range[0] = min_axis_node_idx
                                if max_axis_node_idx > axis_node_range[1]:
                                    axis_node_range[1] = max_axis_node_idx
                        if new_range is True:
                            axis_node_ranges.append(axis_node_range)
            for axis_node_range in axis_node_ranges:
                # print('Axis node range:', axis_node_range)
                axis_vol += (axis_node_range[1] - axis_node_range[0]) * side_length * (math.pi * (member_radii ** 2))
            axis_vols.append(axis_vol)
        total_axis_vol = sum(axis_vols)

        # prune member volumes
        # print('Total member vol before pruning:', sum(member_volumes))
        pruned_member_volumes = []
        for idx, member_vol in enumerate(member_volumes):
            if idx not in truss_members_vol_ignore:
                pruned_member_volumes.append(member_vol)
        member_volumes = pruned_member_volumes
        # print('Total member volume:', sum(member_volumes))

        # N. print results
        truss_volume = sum(member_volumes) + sum(node_volumes)
        truss_volume -= sum(intersect_vols)
        truss_volume += total_axis_vol

        # Feasibility constraint is based on the number of intersections
        # invert the constraint such that higher is better (1 is feasible)
        feasibility_constraint = 1 - (num_intersections / self.feasibility_constraint_norm)


        # Augment bit interactions such that it only records interactions where the value is less than the key
        interaction_bit_list = []
        for bit, interactions in bit_interactions.items():
            new_interactions = []
            for interaction in interactions:
                if interaction < bit:
                    new_interactions.append(interaction)
            if len(new_interactions) > 0:
                interaction_bit_list.append(bit)


        # Create autoregressive-safe interaction vector
        interaction_vector = [0 for _ in range(config.num_vars)]
        for bit in interaction_bit_list:
            interaction_vector[bit] = 1


        return (truss_volume / total_volume), feasibility_constraint, interaction_vector



    def get_interaction_vec(self, bit_interactions):
        bit_list = [0 for _ in range(config.num_vars)]
        for bit, interactions in bit_interactions.items():
            bit_list[bit] = 1
        bit_str = ''.join([str(x) for x in bit_list])
        return bit_str







if __name__ == '__main__':
    sidenum = 3
    num_vars = TrussFeatures.get_member_count_repeatable(sidenum)

    # bit_str = '101100001010000000010100101100'
    bit_str =   '101100001010001000010100101100'
    bit_list = [int(x) for x in bit_str]





    engine = VolFrac(sidenum, bit_list)
    results = engine.evaluate()
    engine.visualize()

    vol_frac, feasibility_constraint, int_vector = results



    int_vector_str = ''.join([str(x) for x in int_vector])


    print('-----------> Bit list:', bit_str)
    print('---> Bit Interactions:', int_vector_str)
    # print('--> Full Interactions:', engine.get_interaction_vec(bit_interactions))







