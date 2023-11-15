### Multi Task Learning or training

#### The main goal is making robust ML models; however, we are making them more complex and they will take more time to be trained, need more resources. and so on. Everything in this world is a trade-off!


##### In this project we aim to create and develop a router to distribute and sample data is a common task in machine learning, especially when dealing with large datasets or multi-task learning scenarios. The router decides how to allocate data to different tasks or models during training.

I define a "Text Classification using Pre-trained Language Models" project. Therefore, Multi-Task Learning Approach in this scenario will be:

* Use a pre-trained language model like BERT.
* Modify the output layers to include multiple heads, each dedicated to a specific task (e.g., sentiment classification and named entity recognition).
* Train the model on a combined dataset that includes examples from both tasks.
* During training, use a combined loss function that includes the losses for both sentiment classification and named entity recognition.
