def factorial1(n):
    if n == 0:
        return 1
    else:
        return n * factorial1(n - 1)

def sum_factorial_recursive(n):
    if n == 1:
        return factorial1(1)
    else:
        return factorial1(n) + sum_factorial_recursive(n - 1)

n = int(input("Nhập n: "))
sum_factorial = sum_factorial_recursive(n)
print("Tổng S là:", sum_factorial)
