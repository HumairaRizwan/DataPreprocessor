import pyspark
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler,StandardScaler,PCA
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.ml.feature import Imputer




class SmartEncoder():
    
    use_onehot = False 
    use_std_scaler = False
    use_pca = False
   

    def __init__(self, session, df1,pred,use_oversampler,use_onehot, use_std_scaler,use_pca):
        print("Constructor of Class smartencoder ")

        self.session = session
        self.df1 = df1
        self.pred = pred
        self.use_oversampler = use_oversampler
        self.use_onehot = use_onehot
        self.use_std_scaler = use_std_scaler
        self.pca = use_pca
        
    @staticmethod    
    def checkNulls(df):
        null_cols = {col:df.filter(df[col].isNull()).count() for col in df.columns}
        return null_cols
    @staticmethod    
    def checkDuplicates(df):
        if df.count() > df.dropDuplicates().count():
            raise ValueError('Data has duplicates')
    
    
    def handlingMissingVal(self,drops):
        
        #drop cols when the whole row is null
        df= self.df1.na.drop(how = "all")
        #drop duplicates
        df = df.dropDuplicates()
        df = df.drop(*drops)
        #Treating missing values
        cols_to_drop = [x for x in df.columns if df.filter(df[x].isNull()).count()>0]
        if len(cols_to_drop) != 0:
            dff=df.select(cols_to_drop)
            continuousCols =[item[0] for item in dff.dtypes if item[1] != 'string']
            imputer = Imputer(inputCols=continuousCols, outputCols=continuousCols).setStrategy("mean")
            model = imputer.fit(df)
            imputed_data = model.transform(df)
    
        else:
            imputed_data = df
            
        return imputed_data
    
    def indexString(self,drops):
        
        imputed = self.handlingMissingVal(drops)
        if self.use_oversampler == True:
                imputed = self.oversmaple(imputed)
        
        #index the string to numeric
        self.stringCols = [item[0] for item in imputed.dtypes if item[1] == 'string']
        if len(self.stringCols) != 0:   
            outputs=[y+"_encoded" for y in self.stringCols]
            stringIndexer = StringIndexer(inputCols=self.stringCols, outputCols=outputs)
            model = stringIndexer.fit(imputed)
            result = model.transform(imputed)
    
            encoded = result.drop(*self.stringCols)        
        
            if self.use_onehot == True:
                #converting categorical attributes into a binary vector
                encoder = OneHotEncoder(dropLast=False, inputCols=outputs, outputCols=[x + "_vec" for x in self.stringCols])
                encoded2 = encoder.fit(encoded).transform(encoded)
    
                encoded = encoded2.drop(*outputs)
            encoded = encoded.withColumnRenamed(self.pred + '_encoded',self.pred)
            encoded = encoded.withColumnRenamed(self.pred + '_vec',self.pred)
        else:
            encoded = imputed
            
        return encoded
    
    def oversmaple(self,df):
        # example of random oversampling to balance the class distribution
        from collections import Counter
        from imblearn.over_sampling import RandomOverSampler
        dfp = df.toPandas()
        # define oversampling strategy
        oversample = RandomOverSampler()
        # fit and apply the transform
        data, y = oversample.fit_resample(dfp.loc[:,dfp.columns!=self.pred], dfp[self.pred])
        data[self.pred] = y
        df_sample = self.session.createDataFrame(data)
        # summarize class distribution
        print(Counter(y))
        
        return df_sample
        
  
    def dataAssembler(self,drops):      
        # VectorAssembler - tranform features into a feature vector column
        encoded = self.indexString(drops)
    
        assembler = VectorAssembler(inputCols = encoded.drop(self.pred).columns, outputCol = 'features')
        df_Assembled = assembler.transform(encoded)
        
        if self.use_std_scaler == True:
            # standardize the dataframe to ensure that all the variables are around the same scale
            scale=StandardScaler(inputCol='features',outputCol='standardized')
            df_scale=scale.fit(df_Assembled)
            df_Assembled=df_scale.transform(df_Assembled)
            
        if self.pca == True:
            pca = PCA(k=3, inputCol=df_Assembled.columns[-1])
            pca.setOutputCol("pca_features")

            model = pca.fit(df_Assembled)
            df_Assembled = model.transform(df_Assembled)
            
            
        return df_Assembled



#test

spark = SparkSession.builder.appName('customer churn').getOrCreate()

df = spark.read.csv('D:\CODE\Raccoon-AI-Engine\datasets\customer_churn.csv', header = True, inferSchema = True)
obj = SmartEncoder(spark,df,'Churn',True,True,True,False)
df_fin = obj.dataAssembler('')
df_fin.show(5)
