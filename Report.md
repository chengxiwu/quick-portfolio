# Summary section

   1. In this project,I mainly focus on the question: as the return value between day t to day t2 and day t3 to day t10 increased, will the trend line of positve socre in sentiment analysis increased as well? Or, as the return value between day t to day t2 and day t3 to day t10 decreased, will the trend line of negative socre in sentiment analysis decreased as well?
    
    
   2. After processing the code, form BHR positive as well as negative and LM positive along with negative, I notice that dspite the trend line on the graph is not that obvious, we still can see slightly difference.

# Data section

Sample we used: 
| Sample used | Description | 
| --- | --- | 
| Sp500 | The S&P 500, or Standard & Poor's 500 | 
|ccm | crsp_2022_only.zip | 
|df| find accession numbers in each firm|
|new_sp500|merge filing accession number to sp500|
|filing_datedf|finding filing date according to accession number|
|sample|merge the return value in each correspond firm|
|cumulative_returns|return values in time span 2 days and 3 days to 10 days|
|dfz|cleaned html content in each firm|
|BHR_positive and BHR_negative|ML_positive_unigram.txt and ML_negative_unigram.txt |
|LM_positive and LM_negative|LM_MasterDictionary_1993-2021 |
|FR_positive and FR_negative|Fincancial risk word list|
|P_positive and p_negative|Profit word list|
|pm_positive and pm_negative| portfolio management word list|
|ccm2|2021_ccm_cleaned.dta|


# Step by step caculate the return value
1. Convert the date and Filing Date columns to datetime objects:

The code converts the date and Filing Date columns to datetime objects using the pd.to_datetime() function.

sample['date'] = pd.to_datetime(sample['date'])
sample['Filing Date'] = pd.to_datetime(sample['Filing Date'])


2. Calculate the Count column using the apply() function:

The code converts the date and Filing Date columns to datetime objects using the pd.to_datetime() function.calculates the Count column using the Pandas apply() function. The Count column represents the number of days since the filing date. If the date is less than the Filing Date, the Count value is set to -2.

sample['Count'] = sample.apply(lambda row: (row['date'] - row['Filing Date']).days if row['date'] >= row['Filing Date'] else -2, axis=1)


3. Filter the data based on the Count value and calculate cumulative returns:

Two new DataFrames, filtered_1 and filtered_2, are created by filtering the sample DataFrame based on the Count column:

filtered_1: Contains rows where Count is between 0 and 2, inclusive.
filtered_2: Contains rows where Count is between 3 and 10, inclusive.
For both filtered_1 and filtered_2, the code groups the data by the Symbol column (i.e., the ticker) and calculates the cumulative return using the ret column. The cumulative return is calculated using the formula: np.prod(1 + x) - 1, where x represents the return values.


filtered_1 = sample[(sample['Count'] >= 0) & (sample['Count'] <= 2)]
cumulative_return1 = filtered_1.groupby('Symbol')['ret'].apply(lambda x: np.prod(1 + x) - 1)
print(cumulative_return1)

filtered_2 = sample[(sample['Count'] >= 3) & (sample['Count'] <= 10)]
cumulative_return2 = filtered_2.groupby('Symbol')['ret'].apply(lambda x: np.prod(1 + x) - 1)
print(cumulative_return2)



4. Create DataFrames for each cumulative return and rename the columns:

The cumulative return values are stored in two separate DataFrames, CR_3_10 and CR_2, with the columns renamed to ret_3_10 and ret_2 for the two time intervals.

CR_3_10 = pd.DataFrame(cumulative_return1).rename(columns={'ret': 'ret_3_10'})
CR_2 = pd.DataFrame(cumulative_return2).rename(columns={'ret': 'ret_2'})



5. Merge the DataFrames and calculate summary statistics:

The two DataFrames, CR_2 and CR_3_10, are merged using the merge() function with an outer join to include all available symbols, ensuring no data is lost during the merge. The resulting DataFrame, cumulative_returns, is then passed to the describe() function to provide summary statistics for the combined data.

cumulative_returns = CR_2.merge(CR_3_10, how='outer', on='Symbol', indicator=True, validate='1:1')
cumulative_returns.describe()

# Step by step caculate the BHR positive or negative, LM postive or negative sentiment score

1. Create a copy of the DataFrame dfz:
df2=dfz.copy()


2. Define the sentiment_analysis() function:

A custom sentiment analysis function called sentiment_analysis() is defined. This function takes two arguments:
text: The text to be analyzed for sentiment.
positive_words: A list of words considered to be positive.
The function tokenizes the input text by splitting it into words and converting each word to lowercase while stripping any trailing punctuation. It then calculates the proportion of positive words in the text by dividing the count of positive words by the total number of words in the text.

def sentiment_analysis(text, positive_words):
    words = [word.lower().strip('.,!?') for word in text.split()]
    positive_word_count = sum([1 for word in words if word in positive_words])
    return positive_word_count / len(words)

3. Apply the sentiment_analysis() function to the cleaned_html column:

The code applies the sentiment_analysis() function to the cleaned_html column of the df2 DataFrame using the apply() function. The BHR_positive variable, which presumably contains a list of positive words, is passed as the second argument to the sentiment_analysis() function.

4. Create a new column BHR_positive in the DataFrame:

The result of the sentiment_analysis() function is stored in a new column called BHR_positive in the df2 DataFrame. This column contains the proportion of positive words for each text in the cleaned_html column.

df2['BHR_positive'] = df2['cleaned_html'].apply(lambda text: sentiment_analysis(text, BHR_positive))


This method was also applied in bewlowed sentiment analysis of LM positive, LM negative and BHR_negative.

# Step by step caculate the Financial Risk positive or negative, Profit postive or negative, Portfolio management postive or negative sentiment score.

It loops through each row in df2 using the iterrows() method. For each row, it extracts the value of the "cleaned_html" column into a variable called sentence.

Next, it performs sentiment analysis on sentence using regular expressions and six pre-defined sentiment dictionaries called fr_positive, fr_negative, p_positive, p_negative, pm_positive, and pm_negative. Each of these dictionaries appears to contain a list of words that are associated with positive or negative sentiment in the financial domain. The NEAR_regex function is used to find the proximity of these words to each other within sentence. The resulting values are then divided by the total number of words in sentence to normalize the sentiment scores.



list_k=[]
list_l=[]
list_z=[]
list_x=[]
list_c=[]
list_v=[]
for index,row in df2.iterrows():
    sentence = row['cleaned_html']
    
    FRP_anlysis=(
    len(re.findall(NEAR_regex(fr_positive,partial=False)
                   ,sentence))
    /
    
    len(sentence.split()))
    
    FRN_anlysis=(len(re.findall(NEAR_regex(fr_negative,partial=False),sentence))/len(sentence.split()))
    
    pp_anlysis=(len(re.findall(NEAR_regex(p_positive,partial=False),sentence))/len(sentence.split()))
    
    pn_anlysis=(len(re.findall(NEAR_regex(p_negative,partial=False),sentence))/len(sentence.split()))
    
    pmp_anlysis=(len(re.findall(NEAR_regex(pm_positive,partial=False),sentence))/len(sentence.split()))
    pmn_anlysis=(len(re.findall(NEAR_regex(pm_negative,partial=False),sentence))/len(sentence.split()))
    
    list_k.append(FRP_anlysis)
    list_l.append(FRN_anlysis)
    list_z.append(pp_anlysis)
    list_x.append(pn_anlysis)
    list_c.append(pmp_anlysis)
    list_v.append(pmn_anlysis)
    
df2['FR_positive']=list_k
df2['FR_negative']=list_l
df2['P_positive']=list_z
df2['p_negative']=list_x
df2["pm_positive"]=list_c
df2["pm_negative"]=list_v





# Why did you choose the three topics you did for the “contextual sentiment” measures?

The reason I choose them is beacuse ,firstly, Profit is an important measure because it is the ultimate goal of any business or investment. The sentiment surrounding a particular asset or market can have a significant impact on its profitability. Secondly, portfolio management is another important measure because it involves the management of a collection of assets to achieve specific investment objectives. Finally, financial risk is a critical measure because it is the likelihood of loss due to various factors such as market volatility, credit risk, interest rate changes, and other factors that could impact the value of an investment





# Show and discuss summary stats of your final analysis sample

![Screenshot 2023-03-24 141248.png](attachment:d5eea2f0-eaf8-4024-ba58-9f5b74ac50c7.png)
![Screenshot 2023-03-24 141339.png](attachment:1f934aef-8e24-4809-b68c-279773a99bbc.png)

Each sentiment analysis score is calculated as the frequency of certain keywords related to the respective financial concept in the company's financial news and press releases. These scores can provide insights into how the market perceives these financial concepts and how they may impact the companies.

# Result

![Screenshot 2023-03-24 141710.png](attachment:abe90c66-aca2-404b-ae60-7831be52c0ff.png)
![Screenshot 2023-03-24 142207.png](attachment:3ab81b66-258d-4ed3-bcfa-3305f2c1de35.png)
![Screenshot 2023-03-24 142232.png](attachment:347e3ad5-207c-40a4-9916-2c71264d4974.png)
![Screenshot 2023-03-24 143923.png](attachment:ec8cd91f-0fef-4daa-b8c1-9353e5029e28.png)
![Screenshot 2023-03-24 142252.png](attachment:ea1ce4c2-7d7b-4b45-9172-17287376a6b1.png)
![Screenshot 2023-03-24 142314.png](attachment:f333bd1f-8c12-4e72-9cf2-18f294458042.png)
![Screenshot 2023-03-24 142336.png](attachment:3cffef24-6ebc-46a1-a6a0-790aa09f9399.png)
![Screenshot 2023-03-24 142354.png](attachment:aceb514e-e075-4488-8ab4-9c42e477cc79.png)
![Screenshot 2023-03-24 142414.png](attachment:02c40f5d-723b-4147-bb21-72fcc136d16f.png)
![Screenshot 2023-03-24 142414.png](attachment:b0bdee37-0bc9-498c-b5d3-ea949b9d21a5.png)


# discussion
1.
from above graph, we conclude that for BHR sentiment score, as for negative score increased, return value for time span 2, or time span 3 to 10, it all keep the almost the same. However as for BHR positive socre increased, the return value for time span 3-10 days is decreasing.

As for LM negative score increased, return value for time span 3-10 days also decreased.For LM positive socre increased, return value for both 2 days and 3-10 days all dcecreased.

2.
Both of them have relationship with return values. However, some of my graph shows that positive score increased may result a decreased return value. This happends may beacause positive sentiment may be correlated with increased investor complacency. When investors become complacent, they may take on more risk than they should, which can lead to higher volatility and lower returns over the long term. 


