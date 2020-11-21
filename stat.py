# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.

import random

nums1 = list(range(52))
nums2 = list(range(52))
nums = nums1 + nums2


total = 0
true = 0

for i in range(500000):
    random.shuffle(nums)
    copy = False
    for j in range(len(nums) - 1):
        if nums[j] == nums[j + 1]:
            copy = True
            break
    if not copy:
        true += 1
    total += 1

print(true/total)
