# Conditional statement (if)
x = 5
if x > 10:
    print("x is greater than 10")
else:
    print(f"x is less than or equal to 10 psát dál")

# For loop with range
print("\nFor loop with range:")
for i in range(1, 6):
    print(i)

# For loop over an array
fruits = ["apple", "banana", "cherry"]
print("\nFor loop over an array:")
for fruit in fruits:
    print(fruit)

# While loop
i = 0
while i < 3:
    print(f"Hello {i+1} times")
    i += 1

# Function
def greet(name):
    print(f"Hello, {name}!")

greet("John")

# F-strings with float formatting
pi = 3.14159265359
print(f"\nPi to two decimal places: {pi:.2f}")