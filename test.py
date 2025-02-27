class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        from collections import deque

        x = abs(x)
        y = abs(y)
        if x == 1 and y == 1:
            return 2
        dp = [[0] * 301] * 301

        combos = deque([(0, 0)])
        level = 0
        while combos:
            level_size = len(combos)
            for _ in range(level_size):
                a, b = combos.popleft()
                if a == x and b == y:
                    return level
                if dp[a][b]:
                    continue
                dp[a][b] = 1

                for horizontal in (-2, 2):
                    for vertical in (-1, 1):
                        if 300 >= a + horizontal >= 0 and 300 >= b + vertical >= 0:
                            combos.append((a + horizontal, b + vertical))
                for horizontal in (-1, 1):
                    for vertical in (-2, 2):
                        if 300 >= a + horizontal >= 0 and 300 >= b + vertical >= 0:
                            combos.append((a + horizontal, b + vertical))
            level += 1

print(Solution().minKnightMoves(2, 112))