import csv
import os

def load_data(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def save_data(file_path, data, fieldnames):
    with open(file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def label_titles(data):
    labeled_data = []
    for idx, row in enumerate(data):
        print(f"\nDescription {idx + 1}:")
        print(row['Description'])
        print("\nCandidate 1:")
        print(row['Candidate 1'])
        print("\nCandidate 2:")
        print(row['Candidate 2'])
        
        choice = input("\nWhich title do you prefer? (1 for Candidate 1, 2 for Candidate 2, or 'both' if they are similar): ").strip().lower()
        
        if choice == '1':
            chosen_title = row['Candidate 1']
        elif choice == '2':
            chosen_title = row['Candidate 2']
        elif choice == 'both':
            chosen_title = 'Both are similar'
        else:
            print("Invalid choice. Skipping this entry.")
            continue
        
        row['Chosen Title'] = chosen_title
        labeled_data.append(row)
    
    return labeled_data

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths relative to the script directory
    input_file = os.path.join(script_dir, 'generated_titles_50.csv')
    output_file = os.path.join(script_dir, 'labeled_titles.csv')
    
    # Load data
    data = load_data(input_file)
    
    # Label titles
    labeled_data = label_titles(data)
    
    # Save labeled data
    fieldnames = data[0].keys() | {'Chosen Title'}
    save_data(output_file, labeled_data, fieldnames)
    
    print(f"\nLabeling complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
