import io
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
#word_tokenize accepts a string as an input, not a file.
stop_words = set(stopwords.words('english'))
filepath = "E:\Sentiment_analysis\data.csv" #KEY IN PATH OF SOURCE FILE
outfile = "E:\Sentiment_analysis\data_filtered.csv" #KEY IN PATH OF THE DESTINATION AND CLEAN TEXT FILE

with open(filepath,encoding="utf8", errors='ignore') as file:
    for cnt,line in enumerate(file):
        words = line.split() #this will split the lines into words
        for r in words:
            if not r in stop_words:
                appendFile = open(outfile, 'a')
                # appendFile.write(ps.stem(r) + " ")
                appendFile.write(r + " ")
                appendFile.close()
        appendFile = open(outfile, 'a') #write the cleaned data.
        appendFile.write("\n")
        appendFile.close()

