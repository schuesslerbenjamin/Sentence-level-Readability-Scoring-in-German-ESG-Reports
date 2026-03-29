########################################################################################################################
echo CLI Interface to Rate the Readability of German Sentences
########################################################################################################################

echo "Please provide the sentence you want to rate (press ENTER without a sentence to finish):"
read input_sentence

# Close if no input
if [ -z "$input_sentence" ]; then
    echo "No sentence provided. Exiting."
    exit 0
fi

echo ""
echo "#####################################################"
echo "Rating the Original Sentence on a Scale from 0 to 1"
echo "#####################################################"
source cli-ARA.sh "$input_sentence" "original"
echo ""

echo ""


source cli.sh