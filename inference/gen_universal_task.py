import config
from itertools import combinations
import random
from copy import deepcopy



def get_all_edge_members():
    lhs_members = [
        (1, 2), (1, 3), (2, 3)
    ]
    rhs_members = [
        (7, 8), (7, 9), (8, 9)
    ]
    bot_members = [
        (1, 4), (1, 7), (4, 7)
    ]
    top_members = [
        (3, 6), (3, 9), (6, 9)
    ]
    return lhs_members + rhs_members + bot_members + top_members



def get_edge_members():
    lhs_members = [
        (1, 2), (1, 3), (2, 3)
    ]
    rhs_members = [
        (7, 8), (7, 9), (8, 9)
    ]

    bot_members = [
        (1, 4), (1, 7), (4, 7)
    ]
    top_members = [
        (3, 6), (3, 9), (6, 9)
    ]

    repeatable_member_pairs = []
    for x in range(len(lhs_members)):
        repeatable_member_pairs.append(
            [lhs_members[x], rhs_members[x]]
        )
        repeatable_member_pairs.append(
            [bot_members[x], top_members[x]]
        )

    # sample a random number of repeatable pairs
    num_repeatable_pairs = random.randint(1, len(repeatable_member_pairs))
    repeatable_pairs = random.sample(repeatable_member_pairs, num_repeatable_pairs)

    all_members = []
    for pair in repeatable_pairs:
        all_members.extend(pair)

    return all_members




def check_feasible_member_set(member_set):
    # for each node in each pair, ensure it is seen in at least one other pair
    for pair in member_set:
        for node in pair:
            found = False
            for other_pair in member_set:
                if node in other_pair and pair != other_pair:
                    found = True
                    break
            if not found:
                return False
    return True



def sample_tasks():
    print('Running')

    max_nodes = 100.0
    all_nodes = [x + 1 for x in range(config.sidenum ** 2)]

    all_node_positions = {}
    idx = 1
    for x in range(config.sidenum):
        for y in range(config.sidenum):
            all_node_positions[idx] = (x / max_nodes, y / max_nodes)
            idx += 1
    print(all_node_positions)

    all_members = list(combinations(all_nodes, 2))
    all_edge_members = get_all_edge_members()
    all_non_edge_members = [x for x in all_members if x not in all_edge_members]
    print(all_members)
    print(all_edge_members)
    print(all_non_edge_members)

    task_lens = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    num_tasks_per_len = 5

    all_tasks = []
    all_task_categories = [x for x in range(len(task_lens))]
    all_task_category_types = [x for x in range(num_tasks_per_len)]

    for task_len in task_lens:
        category_tasks = []
        for _ in range(num_tasks_per_len):
            repeatable_pairs = get_edge_members()
            while len(repeatable_pairs) >= (task_len):
                repeatable_pairs = get_edge_members()

            sample_size = task_len - len(repeatable_pairs)
            members_clipped = deepcopy(all_members)
            for pair in repeatable_pairs:
                members_clipped.remove(pair)

            members_sample = random.sample(members_clipped, sample_size)
            while not check_feasible_member_set(repeatable_pairs + members_sample):
                members_sample = random.sample(members_clipped, sample_size)

            task = repeatable_pairs + members_sample
            # sort task by first number in pair then second
            task = sorted(task, key=lambda x: (x[0], x[1]))

            category_tasks.append(task)
        all_tasks.append(category_tasks)

    # Build all task position vectors
    all_task_positions = []
    for task_category in all_tasks:
        category_positions = []
        for task in task_category:
            task_position_vectors = []
            for pair in task:
                pos_vector = []
                pos_vector.extend(all_node_positions[pair[0]])
                pos_vector.extend(all_node_positions[pair[1]])
                task_position_vectors.append(pos_vector)
            category_positions.append(task_position_vectors)
        all_task_positions.append(category_positions)


    return all_tasks, all_task_positions, all_task_categories, all_task_category_types


























if __name__ == '__main__':
    sample_tasks()



