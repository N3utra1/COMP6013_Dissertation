"""
    This is used to debug files written by the program. It is not needed when running the finished product
"""

import numpy as np 
import os
import glob
files_to_check = glob.glob("D:\\stft-chb-mit\\chb06\\*\\*.npy")



def check_shape():
    # checks if there is a file saved which is not the appropriate shape for the ml model
    for file in files_to_check:
        current_file = np.load(file)
        if not current_file.shape == (17, 3841, 2):
            print(f"bad shape for {file}")
            print(current_file.shape)
        del(current_file)

def check_3rd_dim():
    # check to see if the 3rd dim has data in it (debugging)
    for file in files_to_check:
        current_file = np.load(file)
        if current_file.shape[2] == 0:
            print(f"{file}")
        del(current_file)

def check_2nd_dim_chb06():
    files_to_check = glob.glob("D:\\stft-chb-mit\\chb06\\*\\*.npy")
    for file in files_to_check:
        current_file = np.load(file)
        for x in range(current_file.shape[0]):
            for y in range(current_file.shape[1]):
                print(current_file[x][y])
        del(current_file)

def open_windows():
    import matplotlib.pyplot as plt
    def menu():
        def get_path(path, level):
            if level == 3:
                return path
            indent = "\t" * level
            print(f"\n{indent}path: {path}")
            results = glob.glob(path + "*")
            if not results:
                return path
            
            page_size = 10
            num_pages = (len(results) + page_size - 1) // page_size
            page = 1

            while True:
                print(f"Page {page}/{num_pages}:")
                start_index = (page - 1) * page_size
                end_index = min(page * page_size, len(results))
                for i in range(start_index, end_index):
                    item = results[i].split(os.path.sep)[-1]
                    print(f"{indent}{i + 1}. {item}")

                choice = input("Enter the number to select, 'p' for previous page, 'n' for next page, or 'q' to quit: ").strip()
                if choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= len(results):
                        new_path = results[choice - 1]
                        if not os.path.isdir(new_path):
                            show_plot(new_path)
                        path = get_path(new_path + os.path.sep, level + 1)
                elif choice.lower() == 'p':
                    if page > 1:
                        page -= 1
                elif choice.lower() == 'n':
                    if page < num_pages:
                        page += 1
                elif choice.lower() == 'q':
                    return None

        path = get_path("D:\\stft-chb-mit\\", 0)
        if path:
            print(f"\nSelected path: {path}")
            show_plot(path)
        else:
            print("No file selected.")

    def show_plot(file):
        stft_result = np.load(file)
        plt.specgram(stft_result, cmap='hot', sides='default', mode='default')
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.legend([file])
        plt.show()
    menu()


def method_menu():
    print("Available methods:")
    print("1. check_shape()")
    print("2. check_3rd_dim()")
    print("3. check_2nd_dim_chb06()")
    print("4. open_windows()")
    method_choice = input("Enter the number to select the method, or 'q' to go back to the previous menu: ").strip()
    if method_choice == '1':
        check_shape()
    elif method_choice == '2':
        check_3rd_dim()
    elif method_choice == '3':
        check_2nd_dim_chb06()
    elif method_choice == '4':
        open_windows()
    elif method_choice.lower() == 'q':
        return None
    else:
        print("Invalid choice. Please try again.")
        method_menu()

method_menu()