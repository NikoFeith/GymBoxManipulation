# solver.py

from collections import deque

class Solver:
    """
    Simple BFS-based planner for block rearrangement tasks.

    Each state is represented as a tuple:
    - First N elements: position of each block (-1 = unplaced, 0...K = target index)
    - Last element: currently held block index (-1 = holding nothing)

    Action space:
    - 0 to num_blocks-1: pick block i
    - num_blocks to num_blocks+num_targets-1: place at target j
    """

    def __init__(self, num_blocks, num_targets):
        self.num_blocks = num_blocks
        self.num_targets = num_targets
        self.num_actions = num_blocks + num_targets

    def transition(self, state, action):
        """
        Transition function simulating the result of applying an action.

        Args:
            state (tuple): Current state (block locations and held object)
            action (int): Action index

        Returns:
            tuple: New state after applying the action
        """
        state = list(state)
        holding = state[-1]  # Last entry is held object

        if 0 <= action < self.num_blocks:
            block_idx = action
            # Only pick if not already holding and block is unplaced
            if holding == -1 and state[block_idx] == -1:
                state[-1] = block_idx  # Start holding this block

        elif self.num_blocks <= action < self.num_actions:
            target_idx = action - self.num_blocks
            if holding != -1:
                state[holding] = target_idx  # Place held block at target
                state[-1] = -1  # Now holding nothing

        return tuple(state)

    def solve(self, init_state, goal_state, max_depth=20):
        """
        Runs BFS to find a plan from init_state to goal_state.

        Args:
            init_state (tuple): Initial state
            goal_state (tuple): Desired state
            max_depth (int): Maximum depth to search

        Returns:
            list[int] or None: List of actions (plan), or None if unsolvable
        """
        visited = set()
        queue = deque([(init_state, [])])

        while queue:
            state, path = queue.popleft()

            if state == goal_state:
                return path  # Found a solution

            if len(path) >= max_depth or state in visited:
                continue

            visited.add(state)

            for action in range(self.num_actions):
                next_state = self.transition(state, action)
                queue.append((next_state, path + [action]))

        return None  # No solution found within depth

