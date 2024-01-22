-------------------------------------------
A note on file name(s) of GloVE embeddings.
-------------------------------------------

The GloVE embeddings can be placed anywhere in your system but the file names of GloVE `.txt` files should be same as
the one(s) when they are downloaded. Specifically, the names should be as follows:
  * glove.6B.50d.txt
  * glove.6B.100d.txt
  * glove.6B.200d.txt
  * glove.6B.300d.txt

If the names are altered then the program would not be able to read the files and infer dimensions of GLoVE embeddings

--------------------------------------------------
A note on folder containing the GLoVE embeddings
--------------------------------------------------
Although the program searches for files that matches the names of GLoVE embeddings in the path provided, it is still advised to keep the directory clean,
the directory contain nothing else other than the glove embeddings

-------------------------------------------
How to run?
-------------------------------------------
The file `problem.py` runs as a script and need minor to no modifications besides
  * path_to_csv: Should be provided while instantiating Problem()
  * path_to_glove_300_txt: Should be provided while instantiating Problem()
  * paths_to_glove_embeddings: Should be provided as an argument to Q12()

If you need to wrap this script then follow the steps below for best reproducibility

To see solution(s) for any question in Q1-Q13, do the following
  >>> # create an instance of problem with paths to .csv file and glove.6B.300d.txt
  >>> prob = Problem(path_to_csv = "./Dataset1.csv", path_to_glove_300_txt = "./glove.6B/glove.6B.300d.txt")
  >>> prob.Q1()
  >>> prob.Q2()
  >>> prob.Q3()
  >>> prob.Q4()
  >>> prob.Q5()
  >>> prob.Q6()
  >>> prob.Q7()
  >>> prob.Q8()
  >>> prob.Q9()
  >>> prob.Q10()
  >>> prob.Q11()
  >>> prob.Q12(paths_to_glove_embeddings = "./glove.6B/") 
  >>> prob.Q13()
