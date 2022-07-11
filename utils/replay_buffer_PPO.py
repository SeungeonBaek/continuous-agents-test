import typing
from typing import Tuple
from numpy.typing import NDArray
import numpy as np


class ExperienceMemory():
    def __init__(self, capacity: int):
        self.storage = []
        self.capacity = capacity
        self.next_idx = 0

    def _len(self)-> int:
        return len(self.storage)

    def add(self, data: Tuple)-> None:
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self)-> Tuple[NDArray]:
        state, next_state, reward, action, log_policy, done = [], [], [], [], [], []

        for i in range(self._len()):
            s, next_s, r, a, log_p, d = self.storage[i]

            state.append(np.array(s, copy=False))
            next_state.append(np.array(next_s, copy=False))
            reward.append(np.array(r, copy=False))
            action.append(np.array(a, copy=False))
            log_policy.append(np.array(log_p, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(next_state), np.array(reward), np.array(action), np.array(log_policy), np.array(done)

    def clear(self)-> None:
        self.storage.clear()
        self.next_idx = 0


if __name__ == "__main__":
    memory = ExperienceMemory(30000)
    print(memory._len())
    memory.add((1, 2, 3, [4, 5, 6, 7], 8))
    memory.add((1, 2, 3, [4, 5, 6, 7], 8))
    memory.add((1, 2, 3, [4, 5, 6, 7], 8))
    memory.add((1, 2, 3, [4, 5, 6, 7], 8))

    print(memory.sample(1))