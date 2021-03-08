[image1]: assets/spark_ml_overview.png "image1"

# Spark - Machine Learning
How to deal with ***Big Data***?

Why Learn Spark?
Spark is currently one of the most popular tools for big data analytics. You might have heard of other tools such as Hadoop. Hadoop is a slightly older technology although still in use by some companies. Spark is generally faster than Hadoop, which is why Spark has become more popular over the last few years.

There are many other big data tools and systems, each with its own use case. For example, there are database system like Apache Cassandra and SQL query engines like Presto. But Spark is still one of the most popular tools for analyzing large data sets.

Instead of using the own computer it is easier to use a ***distributed system*** of multiple computers (e.g. hundereds of servers on the Amazon Data Center). Spark is one tool to allow this.

Further info you can find in the first part [park-Big-Data-Analytics](https://github.com/ddhartma/Spark-Big-Data-Analytics.git)

Here is an outline of this session:

## Outline
- [Machine Learning with Spark](#ml_with_Spark)
	- [Feature creation](#feature_creation)
		- [Numeric Feature creation](#fnum_feature_creation)
		- [Text Processing](#text_processing)
	- [Model Training](#model_training)
	- [Hyperparamter Tuning](#hyper_para_tuning)
	- [Capabilities & limitations](#Capabilities_limitations)
	- [Train models at Scale](#Train_models_at_Scale)
	

- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Machine Learning with Spark <a name="ml_with_Spark"></a> 
- Spark supports two MAchine Learning Libraries
	- SparkML (standard ML library, DataFrames API)
	- SparkMLlib (an additive base library, to be removed in the future in Spark 3.2)

- For further details see the [MLlib documentation](https://spark.apache.org/docs/latest/ml-guide.html)

- As Spark DataFrames are close to Python DataFrames, Spark MLlib is close to Scikit-learn packages
- Spark MLlib supports pipelines for stitching together 
	- Data cleaning 
	- Feature engineering steps 
	- Training 
	- Prediction
- Spark handles algorithms that scale linearly with input data size
- In terms of distributed computing there are two different ways to achieve parallelization
	- Data parallelization - Use a large dataset and train the same model on smaller subsets of the data in parallel. The Driver Program acts as a Parameter server for most algorithms where the partial result of each iteration gets combined (model results)
	- Task parallelization - Train many models in parallel on a dataset small enough to fit on a single machine

	![image1]


## Feature creation <a name="feature_creation"></a>
## Numeric Feature creation <a name="fnum_feature_creation"></a>
- Open Jupyter Notebook 'numeric_feature.ipynb'

	```
	from pyspark.sql import SparkSession
	from pyspark.ml.feature import RegexTokenizer, VectorAssembler, Normalizer, StandardScaler
	from pyspark.sql.functions import udf
	from pyspark.sql.types import IntegerType

	import re
	```
	```
	# create a SparkSession: note this step was left out of the screencast
	spark = SparkSession.builder \
		.master("local") \
		.appName("Word Count") \
		.getOrCreate()
	```
	### Read in the Data Set
	```
	stack_overflow_data = 'Train_onetag_small.json'
	df = spark.read.json(stack_overflow_data)
	df.head()

	result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php')
	```
	### Tokenization: 
	Let's split string sentences into separate words.
	Use Sparks [Tokenizer](https://spark.apache.org/docs/latest/ml-features.html#tokenizer)
	- Turn the body into lowercase words, 
	- remove puntuation and special characters (done via pattern="\\W")
	- input column: body
	- output column: words
	```
	regexTokenizer = RegexTokenizer(inputCol="Body", outputCol="words", pattern="\\W")
	df = regexTokenizer.transform(df)
	df.head()

	Result:
	ow(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'])
	```
	count the number of words in each body tag
	```
	body_length = udf(lambda x: len(x), IntegerType())
	df = df.withColumn("BodyLength", body_length(df.words))
	```
	count the number of paragraphs and links in each body tag
	```
	number_of_paragraphs = udf(lambda x: len(re.findall("</p>", x)), IntegerType())
	number_of_links = udf(lambda x: len(re.findall("</a>", x)), IntegerType())
	```
	```
	df = df.withColumn("NumParagraphs", number_of_paragraphs(df.Body))
	df = df.withColumn("NumLinks", number_of_links(df.Body))
	```
	Now we have three more columns:
	- BodyLength=71, 
	- NumParagraphs=2, 
	- NumLinks=0
	```
	df.head(2)

	Result:
	[Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], BodyLength=83, NumParagraphs=2, NumLinks=0),
 	
	...
	]
	```
	### VectorAssembler
	- For many models we need to normalize thepr values, to make sure that they span across the same range
	- Otherwise features with the highest values could dominate the results
	- SparksML offers normalizer, standard scaler and minmax-scaler
	- All these functions require vector rows as an input 
	- Numeric columns must be converted to Spark's vector type
	- We can use VectorAssembler from SparkML for that
	- After that: There is a new column called: NumFeatures=DenseVector([83.0, 2.0, 0.0])
	Combine the body length, number of paragraphs, and number of links columns into a vector
	```
	assembler = VectorAssembler(inputCols=["BodyLength", "NumParagraphs", "NumLinks"], outputCol="NumFeatures")
	df = assembler.transform(df)
	df.head()

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'],  BodyLength=83, NumParagraphs=2, NumLinks=0, NumFeatures=DenseVector([83.0, 2.0, 0.0]))
	```
	### Normalize the Vectors
	- Now let's normalize the Spark Vector NumFeatures
	- We get a new column with ScaledNumFeatures=DenseVector([0.9997, 0.0241, 0.0])
	- Normalizer is a transformer which transforms a vector row to unit norm (vector elements sum up to 1)
	```
	scaler = Normalizer(inputCol="NumFeatures", outputCol="ScaledNumFeatures")
	df = scaler.transform(df)
	df.head()

	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], BodyLength=83, NumParagraphs=2, NumLinks=0, NumFeatures=DenseVector([83.0, 2.0, 0.0]), ScaledNumFeatures=DenseVector([0.9997, 0.0241, 0.0]))
	```
	### Scale the Vectors
	- Standardscaler normalize a feature to have unit standard deviation of 1 and a mean of 0
	```
	scaler2 = StandardScaler(inputCol="NumFeatures", outputCol="ScaledNumFeatures2", withStd=True)
	scalerModel = scaler2.fit(df)
	df = scalerModel.transform(df)

	df.head()
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], BodyLength=83, NumParagraphs=2, NumLinks=0, NumFeatures=DenseVector([83.0, 2.0, 0.0]), ScaledNumFeatures=DenseVector([0.9997, 0.0241, 0.0]), ScaledNumFeatures2=DenseVector([0.4325, 0.7037, 0.0]))
	```

## Text Processing <a name="text_processing"></a>
- Open Jupyter Notebook ```text_processing.ipynb```
	```
	... (libraries etc. same as above)
	```
	### Tokenization
	Let's split string sentences into separate words.
	Use Sparks [Tokenizer](https://spark.apache.org/docs/latest/ml-features.html#tokenizer)
	```
	# split the body text into separate words
	regexTokenizer = RegexTokenizer(inputCol="Body", outputCol="words", pattern="\\W")
	df = regexTokenizer.transform(df)
	df.head()
	```
	### CountVectorizer
	- find the term frequencies of the words --> Sparks's CountVectorizer
	- or even better TFIDF Term Frequency Inverse Document = relative specificity over words
	- Below: vocabSize=1000 (keep top 1000 most common words)
	- Spark stores Word Counts as a Sparse Vector (word 0 --> 4 times, word 1 --> 6 times, etc.)
	
	```
	cv = CountVectorizer(inputCol="words", outputCol="TF", vocabSize=1000)
	cvmodel = cv.fit(df)
	df = cvmodel.transform(df)
	df.take(1)

	Result:
	[Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], TF=SparseVector(1000, {0: 4.0, 1: 6.0, 2: 2.0, 3: 3.0, 5: 2.0, 8: 4.0, 9: 1.0, 15: 1.0, 21: 2.0, 28: 1.0, 31: 1.0, 35: 3.0, 36: 1.0, 43: 2.0, 45: 2.0, 48: 1.0, 51: 1.0, 57: 6.0, 61: 2.0, 71: 1.0, 78: 1.0, 84: 3.0, 86: 1.0, 94: 1.0, 97: 1.0, 99: 1.0, 100: 1.0, 115: 1.0, 147: 2.0, 152: 1.0, 169: 1.0, 241: 1.0, 283: 1.0, 306: 1.0, 350: 2.0, 490: 1.0, 578: 1.0, 759: 1.0, 832: 2.0}))]
	```
	The vocabulary 
	- Most common 1000 words
	- Words close to the beginning are more common than the later ones
	- Here p comes from the \<p> HTML paragraph
	```
	cvmodel.vocabulary

	Result:
	['p',
	'the',
	'i',
	'to',
	'code',
	'a',
	'gt',
	'lt',
	'is',
	'and',
	...
	]
	```
	### Inter-document Frequency
	- If a word appears on a regular base (like 'the', 'a', 'and') it has maybe no deep meaning to get the content 
	- Those stopwords should be less counted 
	- TFIDF is useful for that 
	It surpasses stopwords
	```
	idf = IDF(inputCol="TF", outputCol="TFIDF")
	idfModel = idf.fit(df)
	df = idfModel.transform(df)
	df.head()

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], TF=SparseVector(1000, {0: 4.0, 1: 6.0, 2: 2.0, 3: 3.0, 5: 2.0, 8: 4.0, 9: 1.0, 15: 1.0, 21: 2.0, 28: 1.0, 31: 1.0, 35: 3.0, 36: 1.0, 43: 2.0, 45: 2.0, 48: 1.0, 51: 1.0, 57: 6.0, 61: 2.0, 71: 1.0, 78: 1.0, 84: 3.0, 86: 1.0, 94: 1.0, 97: 1.0, 99: 1.0, 100: 1.0, 115: 1.0, 147: 2.0, 152: 1.0, 169: 1.0, 241: 1.0, 283: 1.0, 306: 1.0, 350: 2.0, 490: 1.0, 578: 1.0, 759: 1.0, 832: 2.0}), TFIDF=SparseVector(1000, {0: 0.0026, 1: 0.7515, 2: 0.1374, 3: 0.3184, 5: 0.3823, 8: 1.0754, 9: 0.3344, 15: 0.5899, 21: 1.8551, 28: 1.1263, 31: 1.1113, 35: 3.3134, 36: 1.2545, 43: 2.3741, 45: 2.3753, 48: 1.2254, 51: 1.1879, 57: 11.0264, 61: 2.8957, 71: 2.1945, 78: 1.6947, 84: 6.5898, 86: 1.6136, 94: 2.3569, 97: 1.8218, 99: 2.6292, 100: 1.9206, 115: 2.3592, 147: 5.4841, 152: 2.1116, 169: 2.6328, 241: 2.5745, 283: 3.2325, 306: 3.2668, 350: 6.2367, 490: 3.8893, 578: 3.6182, 759: 3.7771, 832: 8.8964}))
	```
	### StringIndexer
	- Let's covert the oneTag field that contains strings into numeric values
	- Input column: oneTag
	- Output column: label
	- here php was transformed to label 3.0
	- StringIndexer gives the 0th index the most common string
	- Numeric values (dummy variables) are needed for Sparks ML models both as features and labels
	```
	indexer = StringIndexer(inputCol="oneTag", outputCol="label")
	df = indexer.fit(df).transform(df)
	df.head()

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], TF=SparseVector(1000, {0: 4.0, 1: 6.0, 2: 2.0, 3: 3.0, 5: 2.0, 8: 4.0, 9: 1.0, 15: 1.0, 21: 2.0, 28: 1.0, 31: 1.0, 35: 3.0, 36: 1.0, 43: 2.0, 45: 2.0, 48: 1.0, 51: 1.0, 57: 6.0, 61: 2.0, 71: 1.0, 78: 1.0, 84: 3.0, 86: 1.0, 94: 1.0, 97: 1.0, 99: 1.0, 100: 1.0, 115: 1.0, 147: 2.0, 152: 1.0, 169: 1.0, 241: 1.0, 283: 1.0, 306: 1.0, 350: 2.0, 490: 1.0, 578: 1.0, 759: 1.0, 832: 2.0}), TFIDF=SparseVector(1000, {0: 0.0026, 1: 0.7515, 2: 0.1374, 3: 0.3184, 5: 0.3823, 8: 1.0754, 9: 0.3344, 15: 0.5899, 21: 1.8551, 28: 1.1263, 31: 1.1113, 35: 3.3134, 36: 1.2545, 43: 2.3741, 45: 2.3753, 48: 1.2254, 51: 1.1879, 57: 11.0264, 61: 2.8957, 71: 2.1945, 78: 1.6947, 84: 6.5898, 86: 1.6136, 94: 2.3569, 97: 1.8218, 99: 2.6292, 100: 1.9206, 115: 2.3592, 147: 5.4841, 152: 2.1116, 169: 2.6328, 241: 2.5745, 283: 3.2325, 306: 3.2668, 350: 6.2367, 490: 3.8893, 578: 3.6182, 759: 3.7771, 832: 8.8964}), label=3.0)


	```

















## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.



### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Spark-Big-Data-Analytics.git
```

- Change Directory
```
$ cd Spark-Big-Data-Analytics
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
pyspark
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
