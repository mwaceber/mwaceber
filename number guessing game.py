import random

number_guessed = random.randint(1, 100)
while True:
    try :
        player_choice = int(input("Guess the number between 1 and 100: "))

        if player_choice < number_guessed :
            print("A bit low lad! Try again.")
        elif player_choice > number_guessed :
            print("A bit high lad! Try again.")
        else :
            print("Congratulations. You guessed it right mate.!")
            break
    except ValueError :
        print("Invalid choice. Try again!")