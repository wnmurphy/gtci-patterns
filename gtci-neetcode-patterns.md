# Grokking Technical Interview Patterns - Study sheet

## 1. Two Pointers

Use for problems involving arrays or linked lists where you need to find pairs that meet certain criteria.

```python
def two_pointers(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]
```

### Practice problems

* [125 - Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
* [15 - 3Sum](https://leetcode.com/problems/3sum/)
* [19 - Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
* [75 - Sort Colors](https://leetcode.com/problems/sort-colors/)
* [151 - Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/)
* [680 - Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)


## 2. Fast and Slow Pointers

Use for problems involving cycles in linked lists or arrays.

```python
def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### Practice problems

* [202 - Happy Number](https://leetcode.com/problems/happy-number/)
* [141 - Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
* [876 - Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
* [457 - Circular Array Loop](https://leetcode.com/problems/circular-array-loop/)
* [287 - Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
* [234 - Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)


## 3. Sliding Window

Use for problems involving arrays or strings where you need to find a subarray or substring that meets certain conditions.

```python
def sliding_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

### Practice problems

* [187 - Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/)
* [239 - Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
* [76 - Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
* [424 - Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
* [76 - Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
* [3 - Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
* [209 - Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
* [121 - Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)


## 4. Merge Intervals

Use for problems involving intervals or ranges.

```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], intervals[i][1])
        else:
            merged.append(intervals[i])
    return merged
```

### Practice problems

* [56 - Merge Intervals](https://leetcode.com/problems/merge-intervals/)
* [57 - Insert Interval](https://leetcode.com/problems/insert-interval/)
* [986 - Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)
* [759 - Employee Free Time](https://leetcode.com/problems/employee-free-time/)
* [621 - Task Scheduler](https://leetcode.com/problems/task-scheduler/)
* [253 - Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)


## 5. In-Place Manipulation of a Linked List

Use for problems where you need to reverse a portion or the entirety of a linked list without using extra space. Common scenarios include reversing the entire list, reversing sublists, or rearranging the nodes in a specific order.

```python
# Template
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def reverse_linked_list(head):
    prev, curr = None, head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

```python
def reverse_sublist(head, p, q):
    current, prev = head, None
    for _ in range(p - 1):
        prev, current = current, current.next
    last_node_first_part, last_node_sublist = prev, current
    for _ in range(q - p + 1):
        temp = current.next
        current.next = prev
        prev, current = current, temp
    if last_node_first_part:
        last_node_first_part.next = prev
    else:
        head = prev
    last_node_sublist.next = current
    return head
```

### Practice problems

* [206 - Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
* [25 - Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)
* [92 - Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)
* [143 - Reorder List](https://leetcode.com/problems/reorder-list/)
* [1721 - Swapping Nodes in a Linked List](https://leetcode.com/problems/swapping-nodes-in-a-linked-list/)
* [2074 - Reverse Nodes in Even Length Groups](https://leetcode.com/problems/reverse-nodes-in-even-length-groups/)
* [24 - Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)


## 6. Two Heaps

Use for problems involving balancing two halves of data, often to find medians efficiently, as finding the median of a data stream.

```python
from heapq import heappush, heappop

class MedianOfAStream:
    def __init__(self):
        self.max_heap = []  # containing first half of numbers
        self.min_heap = []  # containing second half of numbers

    def insert_num(self, num):
        if not self.max_heap or num <= -self.max_heap[0]:
            heappush(self.max_heap, -num)
        else:
            heappush(self.min_heap, num)

        # Balance the heaps
        if len(self.max_heap) > len(self.min_heap) + 1:
            heappush(self.min_heap, -heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heappush(self.max_heap, -heappop(self.min_heap))

    def find_median(self):
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        return -self.max_heap[0]
```

### Practice problems

* [502 - IPO](https://leetcode.com/problems/ipo/)
* [295 - Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
* [480 - Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)
* [621 - Task Scheduler](https://leetcode.com/problems/task-scheduler/)


## 7. K-way Merge

Use for problems involving merging multiple sorted arrays or lists.

```python
from heapq import heappush, heappop

def merge_k_sorted_lists(lists):
    min_heap, result = [], []
    for i, lst in enumerate(lists):
        if lst:
            heappush(min_heap, (lst[0], i, 0))
    while min_heap:
        val, list_idx, element_idx = heappop(min_heap)
        result.append(val)
        if element_idx + 1 < len(lists[list_idx]):
            heappush(min_heap, (lists[list_idx][element_idx + 1], list_idx, element_idx + 1))
    return result
```

### Practice problems

* [88 - Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)
* [373 - Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)
* [23 - Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
* [378 - Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)


## 8. Top K Elements

Use for problems involving finding the top K elements in a dataset, often using heaps

```python
from heapq import heappush, heappop

def find_k_largest(nums, k):
    min_heap = []
    for num in nums:
        heappush(min_heap, num)
        if len(min_heap) > k:
            heappop(min_heap)
    return list(min_heap)
```

### Practice problems

* [703 - Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)
* [767 - Reorganize String](https://leetcode.com/problems/reorganize-string/)
* [973 - K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)
* [347 - Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
* [215 - Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
* [692 - Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)


## 9. Modified Binary Search

Use for problems involving searching in sorted arrays with modifications.

```python
def binary_search(arr, key):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == key:
            return mid
        elif arr[mid] < key:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### Practice problems

* [704 - Binary Search](https://leetcode.com/problems/binary-search/)
* [33 - Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
* [278 - First Bad Version](https://leetcode.com/problems/first-bad-version/)
* [528 - Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight/)
* [658 - Find K Closest Elements](https://leetcode.com/problems/find-k-closest-elements/)
* [540 - Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/)
* [81 - Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)


## 10. Subsets

Use for problems involving generating subsets or permutations.

```python
def find_subsets(nums):
    subsets = [[]]
    for num in nums:
        subsets += [curr + [num] for curr in subsets]
    return subsets
```

### Practice problems

* [78 - Subsets](https://leetcode.com/problems/subsets/)
* [46 - Permutations](https://leetcode.com/problems/permutations/)
* [17 - Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)
* [22 - Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
* [373 - Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)


## 11. Greedy Techniques

Use for optimization problems where local optimal choices are made at each step with the hope of finding a global optimum. Common scenarios include finding the minimum number of coins for change, scheduling problems, and selecting intervals.


```python
def greedy_algorithm(items):
    # Sort items based on a criterion (e.g., value, cost, ratio)
    items.sort(key=lambda x: x[criterion], reverse=True)

    result = []
    for item in items:
        if is_feasible(result, item):
            result.append(item)

    return result

def is_feasible(result, item):
    # Check if adding the item to the result is feasible
    # This will depend on the specific problem constraints
    return True
```

### Practice problems

* [55 - Jump Game](https://leetcode.com/problems/jump-game/)
* [881 - Boats to Save People](https://leetcode.com/problems/boats-to-save-people/)
* [134 - Gas Station](https://leetcode.com/problems/gas-station/)
* [1029 - Two City Scheduling](https://leetcode.com/problems/two-city-scheduling/)
* [871 - Minimum Number of Refueling Stops](https://leetcode.com/problems/minimum-number-of-refueling-stops/)
* [45 - Jump Game II](https://leetcode.com/problems/jump-game-ii/)


## 12. Backtracking

Use for problems that require exploring all possible solutions to find one or all solutions that satisfy certain constraints. It is commonly applied in puzzles, combinatorial problems, and constraint satisfaction problems like the N-Queens problem, Sudoku, and generating permutations and combinations.

```python
def backtrack(path, options):
    if is_solution(path):
        process_solution(path)
        return

    for option in options:
        if is_valid(option, path):
            path.append(option)
            backtrack(path, options)
            path.pop()
```

### Practice problems

* [51 - N-Queens](https://leetcode.com/problems/n-queens/)
* [79 - Word Search](https://leetcode.com/problems/word-search/)
* [337 - House Robber III](https://leetcode.com/problems/house-robber-iii/)
* [93 - Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)
* [733 - Flood Fill](https://leetcode.com/problems/flood-fill/)
* [37 - Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)
* [473 - Matchsticks to Square](https://leetcode.com/problems/matchsticks-to-square/)


## 13. Dynamic Programming

Use for problems involving optimization and decision-making under constraints.

```python
def knapsack(profits, weights, capacity):
    n = len(profits)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n)]

    for c in range(capacity + 1):
        if weights[0] <= c:
            dp[0][c] = profits[0]

    for i in range(1, n):
        for c in range(1, capacity + 1):
            profit1, profit2 = 0, 0
            if weights[i] <= c:
                profit1 = profits[i] + dp[i - 1][c - weights[i]]
            profit2 = dp[i - 1][c]
            dp[i][c] = max(profit1, profit2)

    return dp[n - 1][capacity]
```

### Practice problems

* [416 - Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
* [322 - Coin Change](https://leetcode.com/problems/coin-change/)
* [198 - House Robber](https://leetcode.com/problems/house-robber/)
* [213 - House Robber II](https://leetcode.com/problems/house-robber-ii/)
* [5 - Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
* [53 - Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
* [70 - Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
* [121 - Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
* [152 - Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)
* [139 - Word Break](https://leetcode.com/problems/word-break/)
* [300 - Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
* [91 - Decode Ways](https://leetcode.com/problems/decode-ways/)
* [1143 - Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
* [64 - Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)
* [62 - Unique Paths](https://leetcode.com/problems/unique-paths/)


## 14. Cyclic Sort

Use for problems involving finding missing numbers or duplicates in an array.

```python
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        if nums[i] != nums[nums[i] - 1]:
            nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]
        else:
            i += 1
    return nums
```

### Practice problems

* [268 - Missing Number](https://leetcode.com/problems/missing-number/)
* [41 - First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
* [442 - Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
* [448 - Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)


## 15. Topological Sort

Use for problems involving ordering of tasks or nodes in a graph.

```python
from collections import deque, defaultdict

def topological_sort(vertices, edges):
    in_degree = {i: 0 for i in range(vertices)}
    graph = defaultdict(list)
    for parent, child in edges:
        graph[parent].append(child)
        in_degree[child] += 1

    sources = deque([k for k, v in in_degree.items() if v == 0])
    sorted_order = []

    while sources:
        vertex = sources.popleft()
        sorted_order.append(vertex)
        for child in graph[vertex]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                sources.append(child)

    return sorted_order if len(sorted_order) == vertices else []
```

### Practice problems

* [207 - Course Schedule](https://leetcode.com/problems/course-schedule/)
* [210 - Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
* [310 - Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/)
* [329 - Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)
* [802 - Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states/)


## 16. Matrices

```python
def solve_matrix_problem(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    result = []
    # Iterate over the matrix
    for r in range(rows):
        for c in range(cols):
            # Apply the logic for each cell
            pass
    return result
```

```python
# BFS on a matrix
num_rows, num_cols = len(grid), len(grid[0])
def get_neighbors(coord):
    row, col = coord
    delta_row = [-1, 0, 1, 0]
    delta_col = [0, 1, 0, -1]
    res = []
    for i in range(len(delta_row)):
        neighbor_row = row + delta_row[i]
        neighbor_col = col + delta_col[i]
        if 0 <= neighbor_row < num_rows and 0 <= neighbor_col < num_cols:
            res.append((neighbor_row, neighbor_col))
    return res

from collections import deque

def bfs(starting_node):
    queue = deque([starting_node])
    visited = set([starting_node])
    while len(queue) > 0:
        node = queue.popleft()
        for neighbor in get_neighbors(node):
            if neighbor in visited:
                continue
            # Do stuff with the node if required
            # ...
            queue.append(neighbor)
            visited.add(neighbor)
```

### Practice problems

* [73 - Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)
* [48 - Rotate Image](https://leetcode.com/problems/rotate-image/)
* [54 - Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)
* [1706 - Where Will the Ball Fall](https://leetcode.com/problems/where-will-the-ball-fall/)


## 17. Stacks

```python
def mono_stack(insert_entries):
    stack = []
    for entry in insert_entries:
        while stack and stack[-1] <= entry:
            stack.pop()
            # Do something with the popped item here
        stack.append(entry)
```

### Practice problems

* [224 - Basic Calculator](https://leetcode.com/problems/basic-calculator/)
* [1209 - Remove All Adjacent Duplicates in String II](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/)
* [1249 - Minimum Remove to Make Valid Parentheses](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)
* [636 - Exclusive Time of Functions](https://leetcode.com/problems/exclusive-time-of-functions/)
* [341 - Flatten Nested List Iterator](https://leetcode.com/problems/flatten-nested-list-iterator/)
* [232 - Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)
* [20 - Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)


## 18. Graph DFS

Use for any problem where you need to explore all nodes and edges in a depth-first manner.

```python
def dfs(root, visited):
    for neighbor in get_neighbors(root):
        if neighbor in visited:
            continue
        visited.add(neighbor)
        dfs(neighbor, visited)
```

### Practice problems

* [743 - Network Delay Time](https://leetcode.com/problems/network-delay-time/)
* [1376 - Time Needed to Inform All Employees](https://leetcode.com/problems/time-needed-to-inform-all-employees/)
* [133 - Clone Graph](https://leetcode.com/problems/clone-graph/)
* [261 - Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)
* [815 - Bus Routes](https://leetcode.com/problems/bus-routes/)


## 19. Graph BFS

Use for any problem where you need to explore all nodes and edges in a breadth-first manner.

```python
def bfs(root):
    queue = deque([root])
    visited = set([root])
    while len(queue) > 0:
        node = queue.popleft()
        for neighbor in get_neighbors(node):
            if neighbor in visited:
                continue
            queue.append(neighbor)
            visited.add(neighbor)
```

### Practice problems

* [127 - Word Ladder](https://leetcode.com/problems/word-ladder/)
* [133 - Clone Graph](https://leetcode.com/problems/clone-graph/)
* [210 - Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
* [797 - All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/)
* [994 - Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)


## 20. Tree DFS

Use for problems involving depth-first search on a binary tree.

```python
def dfs(root, order="pre"):
    result = []
    def traverse(node):
        if node:
            if order == "pre": result.append(node.value)
            traverse(node.left)
            if order == "in": result.append(node.value)
            traverse(node.right)
            if order == "post": result.append(node.value)
    traverse(root)
    return result
```

### Practice problems

* [114 - Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)
* [543 - Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
* [297 - Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
* [226 - Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
* [124 - Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
* [108 - Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
* [105 - Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
* [199 - Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)
* [236 - Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
* [98 - Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
* [104 - Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
* [230 - Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)


## 21. Tree BFS

Use for problems involving level order traversal of a binary tree.

```python
from collections import deque

def level_order_traversal(root):
    result, queue = [], deque([root])
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.value)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(current_level)
    return result
```

### Practice problems

* [102 - Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
* [103 - Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
* [116 - Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
* [987 - Vertical Order Traversal of a Binary Tree](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)
* [101 - Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
* [127 - Word Ladder](https://leetcode.com/problems/word-ladder/)
* [116 - Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)


## 22. Trie

Use for problems that involve prefix trees or tries to solve problems related to strings.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### Practice problems

* [208 - Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)
* [1268 - Search Suggestions System](https://leetcode.com/problems/search-suggestions-system/)
* [648 - Replace Words](https://leetcode.com/problems/replace-words/)
* [211 - Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)
* [212 - Word Search II](https://leetcode.com/problems/word-search-ii/)
* [386 - Lexicographical Numbers](https://leetcode.com/problems/lexicographical-numbers/)


## 23. Hash Maps

### Practice problems

* [706 - Design HashMap](https://leetcode.com/problems/design-hashmap/)
* [166 - Fraction to Recurring Decimal](https://leetcode.com/problems/fraction-to-recurring-decimal/)
* [359 - Logger Rate Limiter](https://leetcode.com/problems/logger-rate-limiter/)
* [496 - Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
* [205 - Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/)
* [409 - Longest Palindrome](https://leetcode.com/problems/longest-palindrome/)


## 24. Union Find

Use for problems involving the merging of sets and checking whether elements are in the same set

```python
class UnionFind:
    def __init__(self):
        self.id = {}

    def find(self, x):
        y = self.id.get(x, x)
        if y != x:
            self.id[x] = y = self.find(y)
        return y

    def union(self, x, y):
        self.id[self.find(x)] = self.find(y)
```

### Practice problems

* [684 - Redundant Connection](https://leetcode.com/problems/redundant-connection/)
* [200 - Number of Islands](https://leetcode.com/problems/number-of-islands/)
* [947 - Most Stones Removed with Same Row or Column](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/)
* [128 - Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)
* [1970 - Last Day Where You Can Still Cross](https://leetcode.com/problems/last-day-where-you-can-still-cross/)
* [959 - Regions Cut by Slashes](https://leetcode.com/problems/regions-cut-by-slashes/)
* [721 - Accounts Merge](https://leetcode.com/problems/accounts-merge/)
* [928 - Minimize Malware Spread II](https://leetcode.com/problems/minimize-malware-spread-ii/)
* [399 - Evaluate Division](https://leetcode.com/problems/evaluate-division/)


## 25. Bitwise Manipulation

Use for problems involving bit manipulation and properties of XOR.

```python
def find_single_number(arr):
    result = 0
    for num in arr:
        result ^= num
    return result
```

* [389 - Find the Difference](https://leetcode.com/problems/find-the-difference/)
* [1009 - Complement of Base 10 Number](https://leetcode.com/problems/complement-of-base-10-number/)
* [832 - Flipping an Image](https://leetcode.com/problems/flipping-an-image/)
* [136 - Single Number](https://leetcode.com/problems/single-number/)
* [260 - Single Number III](https://leetcode.com/problems/single-number-iii/)
* [271 - Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/)
* [190 - Reverse Bits](https://leetcode.com/problems/reverse-bits/)