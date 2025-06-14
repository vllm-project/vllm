# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
sss = "abaabaaa"
while 'b' in sss:
    sss = sss.replace('b', 'xxx', 1)
    print(sss)
sss = sss.replace('x', 'b')
print(sss)
