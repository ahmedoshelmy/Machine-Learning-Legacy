# Read in the CSV
census_adult = spark.read.csv()("adult_reduced.csv")

# Show the DataFrame
census_adult.show()

# Load the CSV file into a DataFrame
salaries_df = spark.read.csv("salaries.csv", header=True, inferSchema=True)

# Count the total number of rows
row_count = salaries_df.count()
print(f"Total rows: {row_count}")

# Group by company size and calculate the average of salaries
salaries_df.groupBy("company_size").agg({"salary_in_usd": "avg"}).show()
salaries_df.show()

# Average salary for entry level in Canada
CA_jobs = ca_salaries_df.filter(ca_salaries_df["company_location"] == "CA").filter(ca_salaries_df['experience_level']
 == "EN").groupBy().avg("salary_in_usd")

# Show the result
CA_jobs.show()


# Load the dataframe
census_df = spark.read.json("adults.json")

# Filter rows based on age condition
salary_filtered_census = census_df.filter(census_df["age"] > 40)

# Show the result
salary_filtered_census.show()

# Inferring the schema from the CSV file

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Fill in the schema with the columns you need from the exercise instructions
schema = StructType([StructField("age",IntegerType()),
                     StructField("education_num",IntegerType()),
                     StructField("marital_status",StringType()),
                     StructField("occupation",StringType()),
                     StructField("income",StringType()),
                    ])

# Read in the CSV, using the schema you defined above
census_adult = spark.read.csv("adult_reduced_100.csv", sep=',', header=False, schema=schema)

# Print out the schema
census_adult.printSchema()