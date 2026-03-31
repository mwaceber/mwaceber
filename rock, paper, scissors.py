import random

emojis = {'r': '🪨', 's': '✂️', 'p': '📃'}
choices = ('r', 's', 'p')
while True :
    user_choice = input('Rock, paper, scissors (r/p/s): ').lower()
    if user_choice not in choices :
        print("Invalid choice!")
        continue
    else :
        print(f'You chose: {emojis[user_choice]}')
    computer_choice = random.choice(choices)
    print(f'Computer chose: {emojis[computer_choice]}')

    if user_choice == computer_choice:
        print('It is a tie. Care to try again')
    elif (
    user_choice == 'p' and computer_choice == 'r' or\
    user_choice == 's' and computer_choice == 'p' or\
    user_choice == 'r' and computer_choice == 's'):
        print("Congratulations. You won!")
    else :
        print('Sorry. You lose!')

    should_continue = input('Would you like to continue(y/n)?: ').lower()
    if should_continue == 'y' :
        continue
    elif should_continue == 'n':
        print('Thanks for playing.!')
        break
    else :
        print('Invalid choice.Try again.!')