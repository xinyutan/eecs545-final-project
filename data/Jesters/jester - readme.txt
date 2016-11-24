
http://eigentaste.berkeley.edu/dataset/

Dataset 1

Over 4.1 million continuous ratings (-10.00 to +10.00) of 100 jokes from 73,421 users: collected between April 1999 - May 2003

Save to disk, then unzip to obtain Excel files:

jester_dataset_1_1.zip: (3.9MB) Data from 24,983 users who have rated 36 or more jokes, a matrix with dimensions 24983 X 101.
jester_dataset_1_2.zip: (3.6MB) Data from 23,500 users who have rated 36 or more jokes, a matrix with dimensions 23500 X 101.
jester_dataset_1_3.zip: (2.1MB) Data from 24,938 users who have rated between 15 and 35 jokes, a matrix with dimensions 24,938 X 101.
Format:

3 Data files contain anonymous ratings data from 73,421 users.
Data files are in .zip format, when unzipped, they are in Excel (.xls) format
Ratings are real values ranging from -10.00 to +10.00 (the value "99" corresponds to "null" = "not rated").
One row per user
The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
The sub-matrix including only columns {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} is dense. Almost all users have rated those jokes (see discussion of "universal queries" in the above paper).
The text of the jokes can be downloaded here: jester_dataset_1_joke_texts.zip (92KB)

Format:

100 files
Each file has title init_.html, where _ is 1 to 100
The titles correspond to the ID's of the jokes in the Excel files above
