# put the file list in the positional parameters ($1, $2...).
# the files are sorted in alphanumeric order by the shell globbing
set -- COCO_train2014_000000??????.png

n=0
# loop through the files, increasing a counter at each iteration.
for i do
  # every 4th iteration, append the current file to the end of the list
  [ "$(($n % 10))" -eq 0 ] && set -- "$@" "$i"

  # and pop the current file from the head of the list
  shift
  n=$(($n + 1))
done

# now "$@" contains the files that have been appended.
mv -- "$@" ../../masks_test/masks
