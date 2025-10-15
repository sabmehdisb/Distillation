#!/bin/bash

# List of values ​​for parameter j in the paper:
# j_values=(50 200 200 100 200 100 100 50 100)
j_values=(10 10 10 10 10 10 10 10 10)

# Counter to track the list index
index=0

# Loop for each .txt file in the directory
for i in ../resources/forests/DATA_SET/*.txt; do
 rm ${index}.out
  # Retrieve the value of j for this iteration
  j="${j_values[$index]}"

  # Display the value of tree $j and save it to the file
  echo "tree $j" >> ${index}.out

  # Give a unique output name based on the index
  output_name="output_${index}"


  # Run the command with the value of j and redirect the output to the file
  timeout 14400 bash -c "time ./bornAgain ${i} ${output_name} -trees $j -obj 4" >> ${index}.out 2>&1 &

  # Increment the index for the next j
  index=$((index + 1))

  # If the list length is exceeded, start again from the beginning.
  if [ $index -ge ${#j_values[@]} ]; then
    index=0
  fi
done

