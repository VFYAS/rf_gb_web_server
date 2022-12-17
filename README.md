# Web server with interface for RandomForest and GradientBoosting regression algorithms
## Ensembles implementation
Implements regression algorithms Random Forest and Gradient Boosting with mean squared error as criterion and loss function, based on [sklearn](https://github.com/scikit-learn/scikit-learn) and [scipy](https://github.com/scipy/scipy). To trace error values and time spent on learning on train dataset enable:

    trace_loss=True
    trace_time=True
Also allows to trace these values on validation dataset. To do so just enable said parameters and pass validation dataset to <code>fit</code> method of the model.

RandomForestMSE implementation allows training single trees within model in parallel. Use parameter <code>n_jobs</code> to set preferable amount of concurrent threads or set it to -1 for program to decide optimal amount by itself.


## Web server
[![img.png](https://i.postimg.cc/ryf77wNz/img.png)](https://postimg.cc/LhZy1R7M)