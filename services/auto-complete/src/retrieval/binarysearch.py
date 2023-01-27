from bisect import bisect_left


# Check if it contains suffix
def bisect_contains_check(hist_list, prefix):
    try:
        return hist_list[bisect_left(hist_list, prefix)].startswith(prefix)
    except IndexError:
        return False


# Returns the prefix keys
def bisect_list_slice(hist_list, prefix):
    return hist_list[
        bisect_left(hist_list, prefix) : bisect_left(
            hist_list, prefix[:-1] + chr(ord(prefix[-1]) + 1)
        )
    ]
