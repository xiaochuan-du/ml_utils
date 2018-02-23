# Take away from LB

1. For a time series problem, trying to calculate statistics for test set will lead to overfitting. Thus, it is a good idea to separate a label set and a statistic set. 
2. Leveraging labelled data several times with similar X is like data augumentation in structure data.

Eg,
![piu piu's idea](https://i.loli.net/2018/02/20/5a8bfe5495d7e.png)

3. When feature engineering, concat is more desirable than merging, due to performance consideration.

4. Events and dow should be considered together. For example Fri, Sta, Sun also can ben thought as a kind of holiday.

