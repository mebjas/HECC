# df2hc
### DataFrame To Hierarchical Classifiers

#### Classifier = function (DataFrame, Configuration)
**DataFrame** - Pandas dataframe
**Configuration** - Config with information about Heirarchy, thresholds

#### Additional Utilities
 - `Channel` - create data channels for reusing across the flow<br>
 - `VectorizerHub` - Train, Persist, Load named vectorizers to be used across the flow both during training and in live production environments
 - `Ensemble` - Train, Persist, Load ensembles using `ChannelHub` and `VectorizerHub`
 - `OneVsRestClassifier` - for multi class classification easily create OVR on top of Ensembles

