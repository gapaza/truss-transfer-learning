import math

def calculate_edge_angles(node_pos):
    angles = {}
    for i in node_pos:
        for j in node_pos:
            if i < j:  # Avoid calculating twice for the same pair or for a node with itself
                x1, y1 = node_pos[i]
                x2, y2 = node_pos[j]
                angle_rad = math.atan2(y2 - y1, x2 - x1)  # Calculate angle in radians
                angle_deg = math.degrees(angle_rad)  # Convert radians to degrees
                angles[(i, j)] = abs(angle_deg)
    return angles

def get_node_pair_angle(node_pair, edge_angles):
    for key, value in edge_angles.items():
        if node_pair[0] in key and node_pair[1] in key:
            return value
    raise ValueError('Node pair not found in angle dict:', node_pair)

def calculate_bit_angles(bit_connections_list, edge_angles):
    bit_angles = []
    for bit_idx, node_pairs in enumerate(bit_connections_list):
        angle = 0
        for node_pair in node_pairs:
            angle += get_node_pair_angle(node_pair, edge_angles)
        avg_angle = angle / len(node_pairs)
        bit_angles.append(avg_angle)
    return bit_angles

node_positions = {}
idx = 1
for x in range(5):
    for y in range(5):
        node_positions[idx] = (x, y)
        idx += 1


bit_connection_list = [[[1, 2], [21, 22]], [[1, 3], [21, 23]], [[1, 4], [21, 24]], [[1, 5], [21, 25]], [[1, 6], [5, 10]], [[1, 7]], [[1, 8]], [[1, 9]], [[1, 10]], [[1, 11], [5, 15]], [[1, 12]], [[1, 13]], [[1, 14]], [[1, 15]], [[1, 16], [5, 20]], [[1, 17]], [[1, 18]], [[1, 19]], [[1, 20]], [[1, 21], [5, 25]], [[1, 22]], [[1, 23]], [[1, 24]], [[1, 25]], [[2, 3], [22, 23]], [[2, 4], [22, 24]], [[2, 5], [22, 25]], [[2, 6]], [[2, 7]], [[2, 8]], [[2, 9]], [[2, 10]], [[2, 11]], [[2, 12]], [[2, 13]], [[2, 14]], [[2, 15]], [[2, 16]], [[2, 17]], [[2, 18]], [[2, 19]], [[2, 20]], [[2, 21]], [[2, 22]], [[2, 23]], [[2, 24]], [[2, 25]], [[3, 4], [23, 24]], [[3, 5], [23, 25]], [[3, 6]], [[3, 7]], [[3, 8]], [[3, 9]], [[3, 10]], [[3, 11]], [[3, 12]], [[3, 13]], [[3, 14]], [[3, 15]], [[3, 16]], [[3, 17]], [[3, 18]], [[3, 19]], [[3, 20]], [[3, 21]], [[3, 22]], [[3, 23]], [[3, 24]], [[3, 25]], [[4, 5], [24, 25]], [[4, 6]], [[4, 7]], [[4, 8]], [[4, 9]], [[4, 10]], [[4, 11]], [[4, 12]], [[4, 13]], [[4, 14]], [[4, 15]], [[4, 16]], [[4, 17]], [[4, 18]], [[4, 19]], [[4, 20]], [[4, 21]], [[4, 22]], [[4, 23]], [[4, 24]], [[4, 25]], [[5, 6]], [[5, 7]], [[5, 8]], [[5, 9]], [[5, 11]], [[5, 12]], [[5, 13]], [[5, 14]], [[5, 16]], [[5, 17]], [[5, 18]], [[5, 19]], [[5, 21]], [[5, 22]], [[5, 23]], [[5, 24]], [[6, 7]], [[6, 8]], [[6, 9]], [[6, 10]], [[6, 11], [10, 15]], [[6, 12]], [[6, 13]], [[6, 14]], [[6, 15]], [[6, 16], [10, 20]], [[6, 17]], [[6, 18]], [[6, 19]], [[6, 20]], [[6, 21], [10, 25]], [[6, 22]], [[6, 23]], [[6, 24]], [[6, 25]], [[7, 8]], [[7, 9]], [[7, 10]], [[7, 11]], [[7, 12]], [[7, 13]], [[7, 14]], [[7, 15]], [[7, 16]], [[7, 17]], [[7, 18]], [[7, 19]], [[7, 20]], [[7, 21]], [[7, 22]], [[7, 23]], [[7, 24]], [[7, 25]], [[8, 9]], [[8, 10]], [[8, 11]], [[8, 12]], [[8, 13]], [[8, 14]], [[8, 15]], [[8, 16]], [[8, 17]], [[8, 18]], [[8, 19]], [[8, 20]], [[8, 21]], [[8, 22]], [[8, 23]], [[8, 24]], [[8, 25]], [[9, 10]], [[9, 11]], [[9, 12]], [[9, 13]], [[9, 14]], [[9, 15]], [[9, 16]], [[9, 17]], [[9, 18]], [[9, 19]], [[9, 20]], [[9, 21]], [[9, 22]], [[9, 23]], [[9, 24]], [[9, 25]], [[10, 11]], [[10, 12]], [[10, 13]], [[10, 14]], [[10, 16]], [[10, 17]], [[10, 18]], [[10, 19]], [[10, 21]], [[10, 22]], [[10, 23]], [[10, 24]], [[11, 12]], [[11, 13]], [[11, 14]], [[11, 15]], [[11, 16], [15, 20]], [[11, 17]], [[11, 18]], [[11, 19]], [[11, 20]], [[11, 21], [15, 25]], [[11, 22]], [[11, 23]], [[11, 24]], [[11, 25]], [[12, 13]], [[12, 14]], [[12, 15]], [[12, 16]], [[12, 17]], [[12, 18]], [[12, 19]], [[12, 20]], [[12, 21]], [[12, 22]], [[12, 23]], [[12, 24]], [[12, 25]], [[13, 14]], [[13, 15]], [[13, 16]], [[13, 17]], [[13, 18]], [[13, 19]], [[13, 20]], [[13, 21]], [[13, 22]], [[13, 23]], [[13, 24]], [[13, 25]], [[14, 15]], [[14, 16]], [[14, 17]], [[14, 18]], [[14, 19]], [[14, 20]], [[14, 21]], [[14, 22]], [[14, 23]], [[14, 24]], [[14, 25]], [[15, 16]], [[15, 17]], [[15, 18]], [[15, 19]], [[15, 21]], [[15, 22]], [[15, 23]], [[15, 24]], [[16, 17]], [[16, 18]], [[16, 19]], [[16, 20]], [[16, 21], [20, 25]], [[16, 22]], [[16, 23]], [[16, 24]], [[16, 25]], [[17, 18]], [[17, 19]], [[17, 20]], [[17, 21]], [[17, 22]], [[17, 23]], [[17, 24]], [[17, 25]], [[18, 19]], [[18, 20]], [[18, 21]], [[18, 22]], [[18, 23]], [[18, 24]], [[18, 25]], [[19, 20]], [[19, 21]], [[19, 22]], [[19, 23]], [[19, 24]], [[19, 25]], [[20, 21]], [[20, 22]], [[20, 23]], [[20, 24]]]
edge_angles = calculate_edge_angles(node_positions)
bit_angles = calculate_bit_angles(bit_connection_list, edge_angles)


def get_edge_pairs_from_design(design):
    pairs = []
    for i, x in enumerate(design):
        if x == 1:
            pairs.extend(bit_connection_list[i])
    return pairs

def get_bit_from_pair(node_pair):
    for idx, bit_connections in enumerate(bit_connection_list):
        for bit_connection in bit_connections:
            if node_pair[0] in bit_connection and node_pair[1] in bit_connection:
                return idx
    raise ValueError('Bit not found for node pair:', node_pair)

def get_node_pair_angle(node_pair):
    for key, value in edge_angles.items():
        if node_pair[0] in key and node_pair[1] in key:
            return value
    raise ValueError('Node pair not found in angle dict:', node_pair)

if __name__ == '__main__':
    print(edge_angles)
    print(bit_angles)

    # target_ratio = 3.141592654
    # target_ratio = "{:.2f}".format(target_ratio)
    # print(target_ratio)










