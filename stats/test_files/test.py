from stats import analytics

file = "test_file.csv"

def main():
    
    analyze = analytics(file)
    
    analyze.all()
    print("DONE")
    
if __name__ == "__main__":
    main()