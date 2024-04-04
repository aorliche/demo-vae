cd ../notebook
for file in $(ls | grep 'ContrastiveLearningTwo.*\.ipynb'); do
    renamed=$(echo "$file" | sed 's/ContrastiveLearningTwo/DemoVAE/')
    mv $file $renamed
done
