import random

while True:
    choice = input("Roll the dice(y/n): ").lower()
    if choice == "y":
        dice1 = random.randint(1, 6)
        dice2 = random.randint(1, 6)
        print(f" You got {dice1} on the first dice and {dice2} on the second dice")
    elif choice == "n":
        print(" Thanks for your cooperation. Till next time, Bye!")
        break
    else:
        print(" Invalid choice. Try again!")