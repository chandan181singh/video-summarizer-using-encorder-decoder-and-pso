def knapsack_dp(segments, scores, target_length):
    """
    Dynamic programming solution for knapsack problem
    """
    n = len(segments)
    # Create DP table
    dp = [[0 for _ in range(target_length + 1)] for _ in range(n + 1)]
    selected = [[[] for _ in range(target_length + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(target_length + 1):
            start, end = segments[i-1]
            length = end - start
            if length <= w:
                include_value = scores[i-1] + dp[i-1][w-length]
                if include_value > dp[i-1][w]:
                    dp[i][w] = include_value
                    selected[i][w] = selected[i-1][w-length] + [segments[i-1]]
                else:
                    dp[i][w] = dp[i-1][w]
                    selected[i][w] = selected[i-1][w]
            else:
                dp[i][w] = dp[i-1][w]
                selected[i][w] = selected[i-1][w]
    
    # Return selected segments
    return selected[n][target_length]
