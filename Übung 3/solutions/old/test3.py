M = 5
series = list(range(0, 100, 1))
start_indices = []
end_indices = []

for i in range(M, len(series) - M):
    start = i - M
    end = i + M
    start_indices.append(start)
    end_indices.append(end)

print("Start indices:", start_indices)
print("End indices:", end_indices)