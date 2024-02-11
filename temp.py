





# number_generated = 0
# target_number = 20
# window_size = 5
# step = 100 // target_number
# start = 0
# end =  window_size

# while number_generated < target_number:
#     print(f"window {number_generated+1}: {start} -> {end}")
#     start += step 
#     end += step 
#     number_generated += 1
# print("done")


def create_windows(start, end, window_size, num_windows):
    step = (end - start - window_size) / (num_windows - 1)
    windows = [(int(i * step), int(i * step + window_size)) for i in range(num_windows)]
    return windows

# Parameters
start = 0
end = 200906
window_size = (30 * 256)
num_windows = 8145

windows = create_windows(start, end, window_size, num_windows)

for window in windows:
    print(f"Window from {window[0]} to {window[1]}")