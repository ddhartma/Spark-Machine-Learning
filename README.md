[image1]: assets/spark_ml_overview.png "image1"
[image2]: assets/sl_learning_overview.png "image2"
[image3]: assets/ml_pipeline.png "image3"
[image4]: assets/hyper_tuning.png "image4"
[image5]: assets/parameter_grid_evaluator.png "image5"


# Spark - Machine Learning
Why is Spark a powerful tool for **Big Data** Analytics? 

There are many other big data tools and systems, each with its own use case. For example, there are database systems like Hadoop, Apache Cassandra and SQL query engines like Presto. With regard to Hadoop, Spark - a slightly newer and faster technology than Hadoop - is currently one of the most popular tools for big data analytics. 

In case of Big Data it is easier to use a ***distributed system*** of multiple computers (e.g. hundreds of servers on the Amazon Data Center) than using a single machine. Spark is one framework that enables distributed computing.

Further info is provided in the first part [park-Big-Data-Analytics](https://github.com/ddhartma/Spark-Big-Data-Analytics.git)

The outline of this session:

## Outline
- [Machine Learning with Spark](#ml_with_Spark)
	- [Feature creation](#feature_creation)
		- [Numeric Feature creation](#fnum_feature_creation)
		- [Text Processing](#text_processing)
		- [Dimensionality Reduction via PCA](#dim_reduction)
	- [Supervised ML Algorithms](#sl_learning)
		- [Linear Regression](#lin_reg)
		- [Logistic Regression](#log_reg)
	- [Unsupervised ML Algorithms](#unsupervised)
	- [Machine Learning Pipelines](#ml_pipelines)
	- [Model Selection and Tuning](#model_select_tuning)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Machine Learning with Spark <a name="ml_with_Spark"></a> 
- Spark supports two Machine Learning Libraries
	- SparkML (standard ML library, DataFrames API)
	- SparkMLlib (an additive base library, to be removed in the future in Spark 3.2)

- For further details see the [MLlib documentation](https://spark.apache.org/docs/latest/ml-guide.html).

- As Spark DataFrames are close to Python DataFrames, Spark MLlib is close to scikit-learn packages.
- Spark MLlib supports pipelines for stitching together: 
	- Data cleaning 
	- Feature engineering steps 
	- Training 
	- Prediction
- Spark handles algorithms that scale linearly with input data size.
- In terms of distributed computing there are two different ways to achieve parallelization:
	- Data parallelization - Train the ***same model*** on smaller subsets of the Big Data set in parallel. The Driver Program acts as a Parameter server for most algorithms where the partial result of each iteration gets combined (model results).
	- Task parallelization - Train ***many models*** in parallel on a dataset small enough to fit on a single machine.

	![image1]


## Feature creation <a name="feature_creation"></a>
## Numeric Feature creation <a name="fnum_feature_creation"></a>
- Open Jupyter Notebook ```numeric_feature.ipynb```.

	```
	from pyspark.sql import SparkSession
	from pyspark.ml.feature import RegexTokenizer, VectorAssembler, Normalizer, StandardScaler, MinMaxScaler
	from pyspark.sql.functions import udf
	from pyspark.sql.types import IntegerType

	import re
	```
	```
	# Create a SparkSession: note this step was left out of the screencast
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

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php')
	```
	### Tokenization: 
	Let's split string sentences into separate words.
	Use Sparks [Tokenizer](https://spark.apache.org/docs/latest/ml-features.html#tokenizer)
	- Turn the body into lowercase words:
	- Remove punctuation and special characters (done via pattern="\\W")
	- Input column: body
	- Output column: words
	```
	regexTokenizer = RegexTokenizer(inputCol="Body", outputCol="words", pattern="\\W")
	df = regexTokenizer.transform(df)
	df.head()

	Result:
	ow(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'])
	```
	### BodyLength Feature: count the number of words in each body tag
	```
	body_length = udf(lambda x: len(x), IntegerType())
	df = df.withColumn("BodyLength", body_length(df.words))
	```
	Count the number of paragraphs and links in each body tag
	```
	number_of_paragraphs = udf(lambda x: len(re.findall("</p>", x)), IntegerType())
	number_of_links = udf(lambda x: len(re.findall("</a>", x)), IntegerType())
	```
	### NumParagraphs and NumLinks Feature
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
	- For many models we need to normalize the values, to make sure that they span across the same range.
	- Otherwise features with the highest values could dominate the results.
	- SparksML offers normalizer, standard scaler and minmax-scaler.
	- All these functions require vector rows as an input. 
	- Numeric columns must be converted to Spark's vector type.
	- We can use VectorAssembler from SparkML for that.
	- After that: There is a new column called: NumFeatures=DenseVector([83.0, 2.0, 0.0])
	Combine the body length, number of paragraphs, and number of links columns into a vector.
	```
	assembler = VectorAssembler(inputCols=["BodyLength", "NumParagraphs", "NumLinks"], outputCol="NumFeatures")
	df = assembler.transform(df)
	df.head()

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'],  BodyLength=83, NumParagraphs=2, NumLinks=0, NumFeatures=DenseVector([83.0, 2.0, 0.0]))
	```
	### Normalize the Vectors
	- Now let's normalize the Spark Vector NumFeatures.
	- We get a new column with ScaledNumFeatures=DenseVector([0.9997, 0.0241, 0.0]).
	- Normalizer is a transformer which transforms a vector row to ***unit norm - vector elements sum up to 1***.
	```
	scaler = Normalizer(inputCol="NumFeatures", outputCol="ScaledNumFeatures")
	df = scaler.transform(df)
	df.head()

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], BodyLength=83, NumParagraphs=2, NumLinks=0, NumFeatures=DenseVector([83.0, 2.0, 0.0]), ScaledNumFeatures=DenseVector([0.9997, 0.0241, 0.0]))
	```
	### Scale the Vectors
	- Standardscaler normalize a feature to have ***unit standard deviation of 1*** and a ***mean of 0***.
	```
	scaler2 = StandardScaler(inputCol="NumFeatures", outputCol="ScaledNumFeatures2",  withStd=True, withMean=True)
	scalerModel = scaler2.fit(df)
	df = scalerModel.transform(df)
	df.head()

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], BodyLength=83, NumParagraphs=2, NumLinks=0, NumFeatures=DenseVector([83.0, 2.0, 0.0]), ScaledNumFeatures=DenseVector([0.9997, 0.0241, 0.0]), ScaledNumFeatures2=DenseVector([0.4325, 0.7037, 0.0]))
	```
	- MINMAX Scaler
	- ***default Scale: (0,1)***
	```
	scaler3 = MinMaxScaler(inputCol="NumFeatures", outputCol="ScaledNumFeatures3")
	scalerModel = scaler3.fit(df)
	df = scalerModel.transform(df)
	df.head()
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
	# Split the body text into separate words
	regexTokenizer = RegexTokenizer(inputCol="Body", outputCol="words", pattern="\\W")
	df = regexTokenizer.transform(df)
	df.head()
	```
	### CountVectorizer
	- Find the term frequencies of the words --> Sparks's CountVectorizer
	- Or even better: TFIDF Term Frequency Inverse Document = relative specificity over words
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
	The vocabulary: 
	- There are 1000 most common words.
	- Words close to the beginning are more common than the later ones.
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
	- If a word appears on a regular base (like 'the', 'a', 'and') it has (maybe) no deep meaning to get the content. 
	- Those stopwords should be less counted. 
	- TFIDF is useful for that. It surpasses stopwords.
	```
	idf = IDF(inputCol="TF", outputCol="TFIDF")
	idfModel = idf.fit(df)
	df = idfModel.transform(df)
	df.head()

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], TF=SparseVector(1000, {0: 4.0, 1: 6.0, 2: 2.0, 3: 3.0, 5: 2.0, 8: 4.0, 9: 1.0, 15: 1.0, 21: 2.0, 28: 1.0, 31: 1.0, 35: 3.0, 36: 1.0, 43: 2.0, 45: 2.0, 48: 1.0, 51: 1.0, 57: 6.0, 61: 2.0, 71: 1.0, 78: 1.0, 84: 3.0, 86: 1.0, 94: 1.0, 97: 1.0, 99: 1.0, 100: 1.0, 115: 1.0, 147: 2.0, 152: 1.0, 169: 1.0, 241: 1.0, 283: 1.0, 306: 1.0, 350: 2.0, 490: 1.0, 578: 1.0, 759: 1.0, 832: 2.0}), TFIDF=SparseVector(1000, {0: 0.0026, 1: 0.7515, 2: 0.1374, 3: 0.3184, 5: 0.3823, 8: 1.0754, 9: 0.3344, 15: 0.5899, 21: 1.8551, 28: 1.1263, 31: 1.1113, 35: 3.3134, 36: 1.2545, 43: 2.3741, 45: 2.3753, 48: 1.2254, 51: 1.1879, 57: 11.0264, 61: 2.8957, 71: 2.1945, 78: 1.6947, 84: 6.5898, 86: 1.6136, 94: 2.3569, 97: 1.8218, 99: 2.6292, 100: 1.9206, 115: 2.3592, 147: 5.4841, 152: 2.1116, 169: 2.6328, 241: 2.5745, 283: 3.2325, 306: 3.2668, 350: 6.2367, 490: 3.8893, 578: 3.6182, 759: 3.7771, 832: 8.8964}))
	```
	### StringIndexer
	- Let's covert the oneTag field that contains strings into numeric values.
	- Input column: oneTag
	- Output column: label
	- Here php was transformed to label 3.0.
	- StringIndexer gives the 0th index the most common string.
	- Numeric values (dummy variables) are needed for Sparks ML models both as features and labels.
	```
	indexer = StringIndexer(inputCol="oneTag", outputCol="label")
	df = indexer.fit(df).transform(df)
	df.head()

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], TF=SparseVector(1000, {0: 4.0, 1: 6.0, 2: 2.0, 3: 3.0, 5: 2.0, 8: 4.0, 9: 1.0, 15: 1.0, 21: 2.0, 28: 1.0, 31: 1.0, 35: 3.0, 36: 1.0, 43: 2.0, 45: 2.0, 48: 1.0, 51: 1.0, 57: 6.0, 61: 2.0, 71: 1.0, 78: 1.0, 84: 3.0, 86: 1.0, 94: 1.0, 97: 1.0, 99: 1.0, 100: 1.0, 115: 1.0, 147: 2.0, 152: 1.0, 169: 1.0, 241: 1.0, 283: 1.0, 306: 1.0, 350: 2.0, 490: 1.0, 578: 1.0, 759: 1.0, 832: 2.0}), TFIDF=SparseVector(1000, {0: 0.0026, 1: 0.7515, 2: 0.1374, 3: 0.3184, 5: 0.3823, 8: 1.0754, 9: 0.3344, 15: 0.5899, 21: 1.8551, 28: 1.1263, 31: 1.1113, 35: 3.3134, 36: 1.2545, 43: 2.3741, 45: 2.3753, 48: 1.2254, 51: 1.1879, 57: 11.0264, 61: 2.8957, 71: 2.1945, 78: 1.6947, 84: 6.5898, 86: 1.6136, 94: 2.3569, 97: 1.8218, 99: 2.6292, 100: 1.9206, 115: 2.3592, 147: 5.4841, 152: 2.1116, 169: 2.6328, 241: 2.5745, 283: 3.2325, 306: 3.2668, 350: 6.2367, 490: 3.8893, 578: 3.6182, 759: 3.7771, 832: 8.8964}), label=3.0)


	```

## Dimensionality Reduction via PCA <a name="dim_reduction"></a>
- Open Jupyter Notebook ```text_processing.ipynb```.
- Useful to remove correlated features and shrink the feature space.
- Pricipal Component Analysis is one of the most common techniques.
- There is a built-in method in Spark's feature library.
- The Result is a DenseVector.
- PCA works well, if the number of input columns is not too high, otherwise **out-of-memory** errors could occur.
- k=100 means that we want to keep 100 components.
	```
	from pyspark.ml.feature import PCA
	pca = PCA(k=100, inputCol="TFIDF", outputCol="pcaTFIDF")
	model = pca.fit(df)
	df = model.transform(df)

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', words=['p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], TF=SparseVector(1000, {0: 4.0, 1: 6.0, 2: 2.0, 3: 3.0, 5: 2.0, 8: 4.0, 9: 1.0, 15: 1.0, 21: 2.0, 28: 1.0, 31: 1.0, 35: 3.0, 36: 1.0, 43: 2.0, 45: 2.0, 48: 1.0, 51: 1.0, 57: 6.0, 61: 2.0, 71: 1.0, 78: 1.0, 84: 3.0, 86: 1.0, 94: 1.0, 97: 1.0, 99: 1.0, 100: 1.0, 115: 1.0, 147: 2.0, 152: 1.0, 169: 1.0, 241: 1.0, 283: 1.0, 306: 1.0, 350: 2.0, 490: 1.0, 578: 1.0, 759: 1.0, 832: 2.0}), TFIDF=SparseVector(1000, {0: 0.0026, 1: 0.7515, 2: 0.1374, 3: 0.3184, 5: 0.3823, 8: 1.0754, 9: 0.3344, 15: 0.5899, 21: 1.8551, 28: 1.1263, 31: 1.1113, 35: 3.3134, 36: 1.2545, 43: 2.3741, 45: 2.3753, 48: 1.2254, 51: 1.1879, 57: 11.0264, 61: 2.8957, 71: 2.1945, 78: 1.6947, 84: 6.5898, 86: 1.6136, 94: 2.3569, 97: 1.8218, 99: 2.6292, 100: 1.9206, 115: 2.3592, 147: 5.4841, 152: 2.1116, 169: 2.6328, 241: 2.5745, 283: 3.2325, 306: 3.2668, 350: 6.2367, 490: 3.8893, 578: 3.6182, 759: 3.7771, 832: 8.8964}), label=3.0, pcaTFIDF=DenseVector([-0.5291, -0.8217, 0.3129, 0.02, 0.0323, -0.1371, 0.1032, -0.1511, 0.4816, -0.2657, 0.9031, -0.1125, 3.1339, -0.4338, -0.1165, 0.3052, 0.9695, -0.7508, 0.1898, -1.0876, 0.5371, -0.8804, 1.5681, -0.3721, -0.4511, 0.6415, 0.5597, -0.0773, 0.4399, 1.0323, -0.8446, 0.7257, -0.6349, -1.3363, -0.9206, 1.5778, -1.8451, -0.2224, -1.1524, -0.0381, -0.0415, 0.3505, 1.2341, -0.4662, 0.8383, 0.772, 0.7149, -1.0151, 0.148, 0.1278, -0.946, -0.6953, -1.5553, -0.9866, 0.7846, -0.7185, 0.946, 0.6609, -0.0182, 1.3281, -0.4261, -0.6093, -0.8237, -0.5232, -0.5305, -0.4872, 0.1315, 0.8463, -1.1532, -1.2489, 0.3981, -1.4053, 0.4366, -0.931, 0.062, 0.9369, 0.8366, -0.7272, 1.5533, -1.9902, -0.4451, 0.9578, 0.364, -0.3055, -0.9719, -1.1939, 1.1266, -0.3546, 1.6776, 2.1847, -0.0966, -1.6945, -0.9625, -0.7207, 0.4287, -0.6703, 0.7134, 0.2583, -1.692, -0.4525]))
	```
	
## Supervised ML Algorithms <a name="sl_learning"></a>
- If labels are categorical --> **Classification**
- If labels are numeric (and continuous) --> **Regression**
- Spark supports:
	- Binary and multiclass classification such as 
		- Logistic regression
		- RandomForest
		- Gradient boosted trees
		- Support vector machines 
		- Naive Bayes
	- Regression	
		- linear regression
		- generalized linear regression
		- Tree based regressions


	![image2]

## Linear Regression <a name="lin_reg"></a>
- Open Jupyter Notebook ```linear_regression.ipynb```.
	```
	from pyspark.sql import SparkSession
	from pyspark.sql.functions import col, concat, count, lit, udf, avg
	from pyspark.sql.types import IntegerType
	from pyspark.ml.feature import RegexTokenizer, VectorAssembler
	from pyspark.ml.regression import LinearRegression
	```
	```
	spark = SparkSession.builder \
		.master("local") \
		.appName("Creating Features") \
		.getOrCreate()

	stack_overflow_data = 'Train_onetag_small.json'

	df = spark.read.json(stack_overflow_data)
	df.persist()
	```
	### Create Features
	```
	df = df.withColumn("Desc", concat(col("Title"), lit(' '), col("Body")))
	```
	```
	regexTokenizer = RegexTokenizer(inputCol="Desc", outputCol="words", pattern="\\W")
	df = regexTokenizer.transform(df)	
	```
	### Create Feature - **DescLength**
	```
	body_length = udf(lambda x: len(x), IntegerType())
	df = df.withColumn("DescLength", body_length(df.words))
	```
	```
	assembler = VectorAssembler(inputCols=["DescLength"], outputCol="DescVec")
	df = assembler.transform(df)
	```
	### Create Label - **NumTags**
	```
	number_of_tags = udf(lambda x: len(x.split(" ")), IntegerType())
	df = df.withColumn("NumTags", number_of_tags(df.Tags))
	```
	Check if **feature DescLength** correlates with **label NumTags**
	In fact, with longer description there are more tags in the body. DescLength seems to be a good feature for a NumTags prediction.
	```
	df.groupby("NumTags").agg(avg(col("DescLength"))).orderBy("NumTags").show()

	Result:
	+-------+------------------+
	|NumTags|   avg(DescLength)|
	+-------+------------------+
	|      1|143.68776158175783|
	|      2| 162.1539186134137|
	|      3|181.26021064340088|
	|      4|201.46530249110322|
	|      5|227.64375266524522|
	+-------+------------------+
	```
	### Create a Linear Regression Model 
	As there is only one feature, try to find the slope, but not the intercept.
	```
	lr = LinearRegression(maxIter=5, regParam=0.0, fitIntercept=False, solver="normal")
	```
	### Choose the data for modeling
	Choose a DataFrame that has only the target and the one feature
	```
	data = df.select(col("NumTags").alias("label"), col("DescVec").alias("features"))
	data.head()

	Result:
	Row(label=5, features=DenseVector([96.0]))
	```
	### Create and Train the Model
	```
	lrModel = lr.fit(data)
	```
	### Check the results
	```
	lrModel.coefficients

	Result:
	DenseVector([0.0079])
	```
	```
	lr_model.intercept

	Result:
	0.0
	```
	```
	lr_model_summary = lrModel.summary
	```
	```
	lr_model_summary.r2

	Result:
	0.4455149596308462
	```

## Logistic Regression <a name="log_reg"></a>
- Open Jupyter Notebook ```logistic_regression.ipynb```.
- Let's use NumTags (Number of tags) as the label.
- Remember that NumTags correlates with the Length of the description.
- Label is a numeric value (in the example below it is 5).
- TFIDF is a SparseVector.
- Use TFIDF as the feature - so try to prove if you can predict the number of tags with the term interdocument frequency.
- LogisticRegression is a standard Modul from Spark's ml.classification library.
- Result from coefficientMatrix:
	- DenseMatrix with 6 rows and 1000 columns.
	- Values are the intercept vectors.
	```
	from pyspark.sql import SparkSession
	from pyspark.ml.feature import RegexTokenizer, CountVectorizer, IDF, StringIndexer, PCA
	from pyspark.sql.functions import udf, col
	from pyspark.sql.types import IntegerType
	from pyspark.ml.classification import LogisticRegression

	import re
	```
	... see above or notebook (Tokenization)
	```
	number_of_tags = udf(lambda x: len(x.split(" ")), IntegerType())
	df = df.withColumn("NumTags", number_of_tags(df.Tags))

	data2 = df.select(col("NumTags").alias("label"), col("TFIDF").alias("features"))
	data2.head()

	Result:
	Row(label=5, features=SparseVector(1000, {0: 0.0026, 1: 0.7515, 2: 0.1374, 3: 0.3184, 5: 0.3823, 8: 1.0754, 9: 0.3344, 15: 0.5899, 21: 1.8551, 28: 1.1263, 31: 1.1113, 35: 3.3134, 36: 1.2545, 43: 2.3741, 45: 2.3753, 48: 1.2254, 51: 1.1879, 57: 11.0264, 61: 2.8957, 71: 2.1945, 78: 1.6947, 84: 6.5898, 86: 1.6136, 94: 2.3569, 97: 1.8218, 99: 2.6292, 100: 1.9206, 115: 2.3592, 147: 5.4841, 152: 2.1116, 169: 2.6328, 241: 2.5745, 283: 3.2325, 306: 3.2668, 350: 6.2367, 490: 3.8893, 578: 3.6182, 759: 3.7771, 832: 8.8964}))
	```
	```
	lr2 = LogisticRegression(maxIter=10, regParam=0.0)
	```
	```
	lr2Model2 = lr2.fit(data2)
	```
	```
	lr2Model2.coefficientMatrix

	Result:
	DenseMatrix(6, 1000, [-0.024, -0.0001, -0.0003, -0.0002, -0.0, -0.0001, -0.0, -0.0, ..., 0.0143, -0.0057, -0.0037, -0.0073, 0.0016, 0.0037, 0.0003, -0.0116], 1)
	```
	```
	lr2Model2.summary.accuracy

	Result:
	0.31857
	```

## Unsupervised ML Algorithms <a name="unsupervised"></a>
- Open Jupyter Notenbook ```k_means.ipynb```.
- There is no labeled data. 
- There are clusters.
- Clusters are based on similarities between groups.
- In Spark included are clustering algorithms like: 
	- K-means
	- Latent Dirichlet Allocation
	- Gaussian Mixture Model
- Consider that hybrid or semi-supervised (mixture of unsupervised and supervised learning) is not fully implemented in Spark (own implementations are needed).
- Example below: 
	- k-means approach 
	- with five clusters
	- feature col "DescVec"
	- prediction col "DescGroup" 
	```
	from pyspark.sql import SparkSession
	from pyspark.sql.functions import avg, col, concat, count, desc, explode, lit, min, max, split, stddev, udf
	from pyspark.sql.types import IntegerType
	from pyspark.ml.feature import RegexTokenizer, VectorAssembler
	from pyspark.ml.regression import LinearRegression
	from pyspark.ml.clustering import KMeans
	```
	... preprocessing see notebook
	```
	kmeans = KMeans().setParams(featuresCol="DescVec", predictionCol="DescGroup", k=5, seed=42)
	model = kmeans.fit(df)
	df = model.transform(df)
	```
	```
	df.head()

	Result:
	Row(Body="<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", Id=1, Tags='php image-processing file-upload upload mime-types', Title='How to check if an uploaded file is an image without mime type?', oneTag='php', Desc="How to check if an uploaded file is an image without mime type? <p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>\n\n<p>Is there a way to check if the uploaded file is an image apart from checking the file extension using PHP?</p>\n", words=['how', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'without', 'mime', 'type', 'p', 'i', 'd', 'like', 'to', 'check', 'if', 'an', 'uploaded', 'file', 'is', 'an', 'image', 'file', 'e', 'g', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'or', 'another', 'file', 'the', 'problem', 'is', 'that', 'i', 'm', 'using', 'uploadify', 'to', 'upload', 'the', 'files', 'which', 'changes', 'the', 'mime', 'type', 'and', 'gives', 'a', 'text', 'octal', 'or', 'something', 'as', 'the', 'mime', 'type', 'no', 'matter', 'which', 'file', 'type', 'you', 'upload', 'p', 'p', 'is', 'there', 'a', 'way', 'to', 'check', 'if', 'the', 'uploaded', 'file', 'is', 'an', 'image', 'apart', 'from', 'checking', 'the', 'file', 'extension', 'using', 'php', 'p'], DescLength=96, DescVec=DenseVector([96.0]), NumTags=5, DescGroup=0)
	```
	```
	df.groupby("DescGroup").agg(avg(col("DescLength")), avg(col("NumTags")), count(col("DescLength"))).orderBy("avg(DescLength)").show()

	Result:
	+---------+------------------+------------------+-----------------+
	|DescGroup|   avg(DescLength)|      avg(NumTags)|count(DescLength)|
	+---------+------------------+------------------+-----------------+
	|        0| 96.71484436347646|   2.7442441184785|            63674|
	|        4| 241.0267434466191| 3.093549070868367|            28306|
	|        2|499.83863263173606|3.2294372294372296|             6699|
	|        1|      1074.2109375|3.2864583333333335|             1152|
	|        3|2731.0828402366865|  3.42603550295858|              169|
	+---------+------------------+------------------+-----------------+
	```


## Machine Learning Pipelines <a name="ml_pipelines"></a>
- Open Jupyter Notebook ```ml_pipeline.ipynb```.
- Pipelines have two main components. 
- Transformer is an algorithm that transforms a DataFrame to another.
- There are two use cases for transformers: 
	- Feature transformation (tokenizer - changing text --> to numerical values, scaling columns etc.)
	- Making predictions for supervised learning models by transforming features using a model into predicted outcomes
- Estimators fit algorithm parameters on a DataFrame to create a transformer. 
	- It fits or trains data.
	- It implements a method **fit** which accepts a DataFrame and produces a model.
- A Pipeline chains multiple transformers and estimators together to create an ML workflow.

	![image3]
- Spark offers similar pipelines like in scikit-learn.  The chaining process need to be a **directed acyclic graph**. 
	```
	print(type(lr2))

	Result:
	<class 'pyspark.ml.classification.LogisticRegression'>
	```
	```
	# lr2Model2 is a transformer
	print(type(lr2Model2))

	Result:
	<class 'pyspark.ml.classification.LogisticRegressionModel'>
	```
	Create a new dataframe
	```
	df2 = spark.read.json(stack_overflow_data)
	df2.persist()
	```
	- Implement a RegexTokenizer that turns the body into a list of words.
	- Use CountVectorizer to transform the words into term frequencies.
	- Set vocab size to 10000.
	- Use IDF to transform term frequencies into TFIDF.
	- Use Stringindexer to transform string tags into numeric values.
	- Define a LogisticRegression Model.
	- Define the actual pipeline object.
	- As this set of transformers depends on the previous one the chaining order in the pipeline must be kept.
	- StringIndexer is independent from the previous step.
	```
	# split the body text into separate words
	regexTokenizer = RegexTokenizer(inputCol="Body", outputCol="words", pattern="\\W")
	cv = CountVectorizer(inputCol="words", outputCol="TF", vocabSize=10000)
	idf = IDF(inputCol="TF", outputCol="features")
	indexer = StringIndexer(inputCol="oneTag", outputCol="label")

	lr = LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)

	pipeline = Pipeline(stages=[regexTokenizer, cv, idf, indexer, lr])
	```
	Now train the pipeline
	```
	plr = pipeline.fit(df2)
	```
	There are two new columns: 
	- ***probability*** --> type DenseVector
	- ***prediction*** --> type numercal value
	- You can see that the third label (in probability) is quite high with 85%. This is the prediction.
	```
	df3 = plrModel.transform(df2)
	df3.head()

	Result:
	...
	probability=DenseVector([0.027, 0.0083, 0.0077, 0.855, 0.0023, 0.0033, 0.0005, 0.0006, 0.0078, 0.0007, 0.0022, 0.0008, 0.0047, 0.0006, 0.0004, 0.0034, 0.001, 0.0013, 0.001, 0.0036, 0.0006, 0.0007, 0.0007, 0.0006, 0.0004, 0.0011, 0.001, 0.0009, 0.001, 0.0005, 0.0004, 0.0005, 0.0009, 0.0004, 0.0003, 0.0012, 0.0004, 0.0009, 0.0008, 0.0007, 0.0006, 0.0009, 0.0008, 0.0005, 0.0004, 0.0005, 0.0004, 0.0007, 0.0004, 0.0005, 0.0005, 0.0006, 0.0003, 0.0006, 0.0005, 0.0005, 0.0005, 0.0002, 0.0003, 0.0002, 0.0007, 0.0005, 0.0005, 0.0004, 0.0002, 0.0004, 0.0006, 0.0005, 0.0003, 0.0007, 0.0007, 0.0003, 0.0004, 0.0004, 0.0004, 0.0006, 0.0003, 0.0005, 0.0004, 0.0002, 0.0004, 0.0003, 0.0002, 0.0003, 0.0002, 0.0003, 0.0003, 0.0005, 0.0004, 0.0005, 0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0003, 0.0002, 0.0004, 0.0003, 0.0003, 0.0002, 0.0003, 0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0007, 0.0002, 0.0002, 0.0002, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0003, 0.0002, 0.0003, 0.0002, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0003, 0.0003, 0.0002, 0.0004, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0001, 0.0002, 0.0002, 0.0002, 0.0001, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0001, 0.0002, 0.0001, 0.0002, 0.0002, 0.0001, 0.0002, 0.0002, 0.0001, 0.0002, 0.0001, 0.0002, 0.0001, 0.0002, 0.0001, 0.0001, 0.0002, 0.0001, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0001, 0.0]), prediction=3.0)
	```
	Check the prediction: 
	- Filter where the label == prediction  
	- Then: count
	```
	df3.filter(df3.label == df3.prediction).count()

	Result:
	54616
	```
	For 10000 fits - for half of it is the prediction right.
	
## Model Selection and Tuning <a name="model_select_tuning"></a>
- Open Jupyer Notebook ```model_tuning.ipynb```.
- Hyperparameter Tuning is needed to get the best model.
- Spark offers hyperparameter tuning:
	- Train-Validation Split
	- k-fold Cross Validation
- Two parameters are needed:
	1. Define the parameter grid to explore.
	2. Evaluator. It defines the metric to evaluate the results on the test set.
- The best parameter combination is found by using the built-in cross validation methods.
- Spark fits the estimator by using this best parameter set and applying it on the entire dataset.
- Before Spark 2.3 models were trained sequentially. 
- After Spark 2.3 models can be trained in parallel.

	![image5]
	```
	from pyspark.sql import SparkSession
	from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf
	from pyspark.sql.types import IntegerType

	from pyspark.ml import Pipeline
	from pyspark.ml.classification import LogisticRegression
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator
	from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, PCA, RegexTokenizer, StandardScaler, StopWordsRemover, StringIndexer, VectorAssembler
	from pyspark.ml.regression import LinearRegression
	from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

	import re
	```
	Make a random split to create train and test sets.
	```
	train, test = df.randomSplit([0.8, 0.2], seed=42)
	```
	Or use this to take a random split into train, validation and test sets.
	```
	train, rest = df.randomSplit([0.6, 0.4], seed=42)
	test, validation = rest.randomSplit([0.5, 0.5], seed=42)
	```
	Use the pipeline as before.
	```
	regexTokenizer = RegexTokenizer(inputCol="Body", outputCol="words", pattern="\\W")
	cv = CountVectorizer(inputCol="words", outputCol="TF", vocabSize=1000)
	idf = IDF(inputCol="TF", outputCol="features")
	indexer = StringIndexer(inputCol="oneTag", outputCol="label")

	lr =  LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)

	pipeline = Pipeline(stages=[regexTokenizer, cv, idf, indexer, lr])
	```
	Tune the model: Use the k-fold cross validation method
	- Specify the estimator (here pipeline).
	- Specify estimatorParamMaps for a parameter grid search.
	- Specify an evaluator: here the MulticlassClassificationEvaluator
	- numFolds (the part 1/numFolds is for testing, rest for training): e.g. 3 --> keep two thirds for training, use one third for testing. Then repeat this process three times.
	
	For the ParameterGrid:
	- Let's tune the number of words we keep in TFIDF features (e.g. 1000 and 5000 words).
	- Consider: Number of models to train = number of parameters * numFolds
	```
	paramGrid = ParamGridBuilder() \
		.addGrid(cv.vocabSize,[1000, 5000]) \
		.addGrid(lr.regParam,[0.0, 0.1]) \
		.build()


	crossval = CrossValidator(estimator=pipeline,
							estimatorParamMaps=paramGrid,
							evaluator=MulticlassClassificationEvaluator(),
							numFolds=3)
	```
	Check the results: 
	```
	cvModel_q1 = crossval.fit(train)
	```
	For each parameter we get a list as below. 5000 words seem to give a bit better accuracy than 1000 words.
	```
	cvModel_q1.avgMetrics

	Result:
	[0.30653390468824687,
	0.23318033120072945,
	0.3640135546612686,
	0.2865146206452452]

	```
	Check accuracy of the test set via:
	```
	print(results.filter(results.label == results.prediction).count())
	print(results.count())

	Result:
	3892
	9919

	3892 / 9919 = 0.392378263937897
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
