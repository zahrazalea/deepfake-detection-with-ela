import streamlit as st
import pandas as pd
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt # type: ignore
import os
import numpy as np
from keras.models import load_model  # type: ignore
from keras.preprocessing.image import img_to_array  # type: ignore

st.set_page_config(page_title="Title", layout="wide")

left_col, center_col, right_col = st.columns([1, 5, 1])

with center_col:
    st.title("Error-Level Analysis (ELA): Deepfake Detection")
    st.write("A Data Science project by Zahra Azalea")
    
    tab1, tab2, tab3 , tab4,tab5= st.tabs(["About", "User Guide", "Demonstration", "Issues", "Code"])

    with tab1:   
        st.subheader("**Introduction**")

        intro_col1, intro_col2= st.columns(2)
        with intro_col1:
            st.markdown("### What's a Deepfake?")
            st.write("*A type of media involving a person, in which their face or body has been digitally altered, usually for malicious intent. (Oxford Media Press, 2023)*")
            st.image("streamlit pics/deepfake.PNG", caption="Original & Deepfake Image",use_container_width=True)
        with intro_col2:
            st.markdown("### What's an error-level analysis (ELA)?")
            st.write("*An image forensics technique that shows the amount of difference between an image and its resave at a certain error level. White means more change, and black indicates no change.*")
            st.image("streamlit pics/ela.PNG", caption="ELA and Non-ELA Image",use_container_width=True)
        
        st.markdown("---")
        st.markdown("### The Basis of this Project")
        st.markdown("#### Problem Statement")
        st.markdown("""
                - *A deepfake detection model that uses classification to distinguish between real and digitally altered images is essential for protecting individuals and businesses.*
                - *Implementing Error Level Analysis (ELA) is expected to enhance the accuracy of deepfake classification.*
            """)        
        st.write("")
        st.markdown("#### Objectives")
        st.markdown("""
                - *To compare ELA methods with non-ELA methods to demonstrate its effectiveness in classifying real and deepfake images.*
                - *To classify real and deepfake images using machine learning.*
                - *To deploy a working product that classifies real and deepfake images.*
            """)

    with tab2:

        st.header("User Guide")

        st.markdown("### Step-by-step")

        st.info("1. Select the 'Demonstration Tab'.")
        st.info("2. Click the 'Browse files' button to upload an image from local files.")
        st.image("streamlit pics/ug1.PNG",use_container_width=True)
        st.markdown("---")
        st.info("3. The uploaded image will be displayed with its original size and resolution.")
        st.image("streamlit pics/ug2.PNG",use_container_width=True)
        st.markdown("---")
        st.info("4. Select 'Yes' or 'No' for ELA implementation.")
        st.image("streamlit pics/ug3.PNG",use_container_width=True)
        st.markdown("---")
        st.info("5. Select the preferred model, a recommended model will be shown below it.")
        st.image("streamlit pics/ug4.PNG",use_container_width=True)
        st.markdown("---")
        st.info("6. Click 'Run Model' to begin analysis.")
        st.image("streamlit pics/ug5.PNG",use_container_width=True)
        st.markdown("---")
        st.info("7. A processed image will be displayed, as well as the classification result and its confidence level.")
        st.image("streamlit pics/ug6.PNG",use_container_width=True)
        
        
    def preprocess(image_pil, bool):
            im = image_pil.convert('RGB')

            if im.size != (224, 224):
                im = im.resize((224, 224))

            if bool == True:
                im_temp = 'temp.jpg'
                im.save(im_temp, 'JPEG', quality=90)
                resaved = Image.open(im_temp)

                diff = ImageChops.difference(im, resaved)
                extrema = diff.getextrema()
                max_px = max([ex[1] for ex in extrema])
                scale_factor = 255.0 / max_px if max_px != 0 else 1
                im = ImageEnhance.Brightness(diff).enhance(scale_factor)
                os.remove(im_temp)

            return im
        
    def predict_image(model, img_path, bool, class_names=["Real", "Deepfake"]):
        if bool == True:
            img = preprocess(img_path, True)
        else:
            img = img_path
        img_resized = img.resize((224, 224))  
        img_array = img_to_array(img_resized)  
        img_array = img_array.astype('float32') / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  
            
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)
        predicted_label = class_names[predicted_class[0]]
        confidence = predictions[0][predicted_class[0]]  
            
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}")
        plt.show()
    
        return predicted_label, confidence


    with tab3:

        st.header("Model Testing")

        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            con_im1,con_im2,con_im3 = st.columns(3)
            with con_im2: st.image(image, caption="Uploaded Image", use_container_width=True)
            
            use_ela = st.selectbox(
                    "Would you like to use error-level analysis (ELA)?",
                    ["Yes", "No"]
                )
   
            model_paths_ela = {
                "DenseNet-121": {
                        "model_path": "models/densenet_ela_model.h5",
                    },
                "InceptionV3": {
                        "model_path": "models/inceptionv3_ela_model.h5",
                    },
                "Xception": {
                        "model_path": "models/xception_ela_model.h5",
                    }
            }

            model_paths = {
                "DenseNet-121": {
                        "model_path": "models/densenet_model.h5",
                    },
                "InceptionV3": {
                        "model_path": "models/inceptionv3_model.h5",
                    },
                "Xception": {
                        "model_path": "models/xception_model.h5",
                    }
            }
            
            if use_ela == "Yes":
                model_options = list(model_paths_ela.keys())  
                chosen_model = st.selectbox("Select your preferred model:", model_options)
                st.success("Recommended Model for ELA Image Analysis: InceptionV3")
                model_path = model_paths_ela[chosen_model]["model_path"]  
            else:
                model_options = list(model_paths.keys())
                chosen_model = st.selectbox("Select your preferred model:", model_options)
                st.success("Recommended Model for Non-ELA Image Analysis: DenseNet-121")
                model_path = model_paths[chosen_model]["model_path"]
                                
            try:
                model = load_model(model_path)                
                if st.button("Run Now"):
                    with st.spinner("Running..."):
                        try:
                            con_im1, con_im2, con_im3 = st.columns(3)
                            with con_im2:
                                if use_ela == "Yes":
                                    bool_im = True
                                    st.image(preprocess(image, True), caption='Image Processed with ELA', use_container_width=True)
                                    st.markdown("#### Results (Using ELA):")
                                else:
                                    bool_im = False
                                    st.image(preprocess(image, False), caption='Image Processed without ELA', use_container_width=True)
                                    st.markdown("#### Results (Without ELA):")
                                
                                predicted_label, confidence = predict_image(model, image, bool_im)
                                st.write(f"Prediction: {predicted_label}")
                                st.write(f"Confidence: {confidence:.2f}")
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
            except Exception as e:
                st.error(f"Error loading {model_path}: {str(e)}")
    
    with tab4:
        st.header("Issues")

        st.markdown("### Model is not reliable. For example,")
        st.markdown('''
                The following tables represent results from testing model using a dataset separate from the one used to train the model (https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) and also using a subset of the dataset
                ''')
        con_im1, con_im2, con_im3,con_im4 = st.columns(4)
        with con_im1: st.image("streamlit pics/newresult.png",caption="Tested using different dataset (ELA)",use_container_width=True)
        with con_im2: st.image("streamlit pics/newresult2.png",caption="Tested using different dataset (Non-ELA)",use_container_width=True)
        with con_im3: st.image("streamlit pics/newresult3.png",caption="Tested using subset of dataset (ELA)",use_container_width=True)
        with con_im4: st.image("streamlit pics/newresult4.png",caption="Tested using subset of dataset (Non-ELA)",use_container_width=True)
        st.write("")
        st.markdown('''
                    #### From the above tables, it is seen that the model does not predict well. Multiple solutions have been tried:

                    - **Cross-validation**: Did not improve performance
                    - **Ensemble method (soft voting, hard voting)**: Did not improve performance
                    - **Unfreezing base model or increase complexity of model**: Worsened performance
                    - **Apply regularization**: Worsened performance
                    - **Tried different loss functions and learning rates**: Current model parameters are the best

                    #### Although various solutions have been applied, it is highly likely that the root problem is related to the dataset used to train the model. Why?
                    ''')
        
        st.markdown('#### Here are examples of proper ELA:')
        con_ela1, con_ela2, con_ela3 ,con_ela4= st.columns(4)
        with con_ela1: st.image("streamlit pics/ela_issue1.PNG",caption='Normal Image (Deepfake)', use_container_width=True)
        with con_ela2: st.image("streamlit pics/ela_issue2.PNG",caption='ELA Image (Deepfake)', use_container_width=True)

        con_ela5, con_ela6, con_ela7 ,con_ela8= st.columns(4)
        with con_ela5: st.image("streamlit pics/ela_issue3.PNG",caption='Normal Image (Real)', use_container_width=True)
        with con_ela6: st.image("streamlit pics/ela_issue4.PNG",caption='ELA Image (Real)', use_container_width=True)

        st.markdown('#### Here are ELA images from the dataset used:')
        con_ela9, con_ela10, con_ela11 ,con_ela12= st.columns(4)
        with con_ela9: st.image("streamlit pics/ela_issue5.png",caption='Normal Image (Deepfake)', use_container_width=True)
        with con_ela10: st.image("streamlit pics/ela_issue6.png",caption='ELA Image (Deepfake)', use_container_width=True)

        con_ela13, con_ela14, con_ela15 ,con_ela16= st.columns(4)
        with con_ela13: st.image("streamlit pics/ela_issue7.png",caption='Normal Image (Real)', use_container_width=True)
        with con_ela14: st.image("streamlit pics/ela_issue8.png",caption='ELA Image (Real)', use_container_width=True)

        st.markdown('### Based on the sample images from the dataset used, there are no clear distinctions of edited areas.')
        st.markdown('#### How does this affect the model performance?')
        st.markdown('''
                    - Images from real and fake dataset both look similar, even after processed with ELA.
                    - Processed and compressed images reduce fine details of image.
                    - Model cannot differentiate between real and fake images, with or without ELA.
                    ''')
        st.markdown('#### Then why is there a difference between performance for images with ELA and images without ELA?')
        con_ela17, con_ela18, con_ela19 = st.columns(3)
        with con_ela18: st.image("streamlit pics/modelresult.PNG",caption='Model Results', use_container_width=True)
        st.markdown('''
                    - Reduced fine details decreases effectiveness of ELA.
                    - Images resaved multiple times also decreases effectiveness of ELA.
                    - Rendering ELA visibility in images to disappear (turn black or near the original image compression level)                   
                    ''')
        
        st.markdown('### How to solve this issue?')
        st.markdown('''
                    - Use a dataset without processed/compressed images.
                    - Do not resize during preprocessing.
                    ''')            


    with tab5:
        st.header("Code")

        with st.container():
            st.subheader("Source Code")
            code_type = st.selectbox(
                "Choose which code to display:",
                ["Import", "Preprocessing", "EDA", "Model", "Evaluation"]
            )

            code_snippets = {
                "Import": {
                    "title":"_Library imports and data loading:_",
                    "code": """
                            import os
                            from PIL import Image, ImageChops, ImageEnhance
                            import numpy as np
                            from tensorflow.keras.applications import DenseNet121,InceptionV3,Xception,MobileNetV3Small,VGG16
                            from tensorflow import keras
                            from keras import layers
                            from tensorflow.keras import layers, models
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            from sklearn.metrics import confusion_matrix, classification_report
                            import pandas as pd
                            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                            from tensorflow.keras.utils import img_to_array

                            train_fake = 'dataset/train/Fake'
                            train_real = 'dataset/train/Real'

                            train_dir = [train_real, train_fake] 

                            val_fake = 'dataset/val/Fake'
                            val_real = 'dataset/val/Real'

                            val_dir = [val_real, val_fake]

                            test_fake = 'dataset/test/Fake'
                            test_real = 'dataset/test/Real'

                            test_dir = [test_real, test_fake]

                            """,
                },
                
                "Preprocessing": {
                    "title":"_Preprocessing for the images before modelling:_",
                    "code": """
                            def ela(im_path):

                                im = Image.open(im_path).convert('RGB')

                                if im.size != (224, 224):
                                    im = im.resize((224, 224))

                                im_temp = 'temp.jpg'
                                im.save(im_temp, 'JPEG', quality=90)
                                resaved = Image.open(im_temp)

                                diff = ImageChops.difference(im, resaved)
                                extrema = diff.getextrema()
                                max_px = max([ex[1] for ex in extrema])
                                scale_factor = 255.0 / max_px if max_px != 0 else 1
                                diff = ImageEnhance.Brightness(diff).enhance(scale_factor)

                                os.remove(im_temp)
                                return diff

                            def labeler(im_dir, ela_bool):
                                X, Y = [], []
                                for dir_path in im_dir:
                                    for name in os.listdir(dir_path):
                                        file_path = os.path.join(dir_path, name)
                                        if ela_bool:
                                            im = ela(file_path)
                                        else:
                                            im = Image.open(file_path).convert('RGB').resize((224, 224))
                                        if name.startswith("real"):
                                            X.append(im)
                                            Y.append(1)
                                        elif name.startswith("fake"):
                                            X.append(im)
                                            Y.append(0)
                                return np.array(X).astype('float32') / 255.0, np.array(Y).astype('int')
                            
                            X_train_ela, Y_train_ela = labeler(train_dir, True)
                            X_val_ela, Y_val_ela = labeler(val_dir, True)
                            X_test_ela, Y_test_ela = labeler(test_dir, True)
                            X_train, Y_train = labeler(train_dir, False)
                            X_val, Y_val = labeler(val_dir, False)
                            X_test, Y_test = labeler(test_dir, False)

                            def shuffler(arr1, arr2): # function to shuffle array to avoid bias
                                shuffle = np.arange(arr1.shape[0])
                                np.random.shuffle(shuffle)
                                return arr1[shuffle], arr2[shuffle]
                            
                            X_test_ela, Y_test_ela = shuffler(X_test_ela, Y_test_ela)
                            X_val_ela, Y_val_ela = shuffler(X_val_ela, Y_val_ela)
                            X_train_ela, Y_train_ela = shuffler(X_train_ela, Y_train_ela)
                            X_test, Y_test = shuffler(X_test, Y_test)
                            X_val, Y_val = shuffler(X_val, Y_val)
                            X_train, Y_train = shuffler(X_train, Y_train)

                            """,
                },
                
                "EDA": {
                    "title":"_Exploring and visualizing prepared data:_",
                    "code": """
                            def display_samples(X_train, Y_train, num_samples):
                                plt.figure(figsize=(num_samples, 4))
                                for i in range(num_samples):
                                    plt.subplot(2, (num_samples + 1) // 2, i + 1)
                                    plt.imshow(X_train[i])  
                                    plt.title(f"Label: {Y_train[i]}")
                                    plt.axis("off")
                                plt.show()

                            display_samples(X_train, Y_train, num_samples=10)
                            display_samples(X_test, Y_test, num_samples=10)
                            display_samples(X_val, Y_val, num_samples=10)
                            display_samples(X_train_ela, Y_train_ela, num_samples=10)
                            display_samples(X_test_ela, Y_test_ela, num_samples=10)
                            display_samples(X_val_ela, Y_val_ela, num_samples=10)

                            def check_class_distribution(Y, title):
                                
                                Y = pd.Series(Y).replace({0: "Fake", 1: "Real"})
                                distribution = Y.value_counts()

                                plt.figure(figsize=(8, 6))
                                distribution.plot(kind='bar', color='skyblue', edgecolor='black')
                                plt.title(title)
                                plt.xlabel("Class")
                                plt.ylabel("Count")
                                plt.xticks(rotation=0)
                                plt.ylim(0, 4000)
                                plt.show()
                            
                            check_class_distribution(Y_train,"Train Set: Class Distribution")
                            check_class_distribution(Y_test,"Test Set: Class Distribution")
                            check_class_distribution(Y_val,"Validation Set: Class Distribution")
                            check_class_distribution(Y_train_ela,"Train Set (ELA): Class Distribution")
                            check_class_distribution(Y_test_ela,"Test Set (ELA): Class Distribution")
                            check_class_distribution(Y_val_ela,"Validation Set (ELA): Class Distribution")

                            """,
                },
                                            
                "Model": {
                    "title":"_CNN model implementation for deepfake detection:_",
                    "code": """
                            
                            def densenet_model_build():
                                base_model = DenseNet121(
                                    include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3)
                                )
                                
                                base_model.trainable = False


                                model = models.Sequential([
                                    layers.Input(shape=(224, 224, 3)),
                                    base_model,
                                    layers.GlobalAveragePooling2D(), #dont put dropout after pooling
                                    layers.Dense(1024,activation='relu'), #best 1024, dont put regularization
                                    layers.Dropout(0.3),
                                    layers.Dense(2, activation='softmax')
                                ])

                                optimizer = keras.optimizers.Adam(5e-5)

                                model.compile(
                                    optimizer=optimizer,
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                return model

                            densenet_ela_model = densenet_model_build()
                            densenet_model = densenet_model_build()

                            lr_scheduler = ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.5,
                                patience=2,
                                verbose=1
                            )

                            early_stopping = EarlyStopping(
                                monitor='val_loss',
                                patience=4,
                                restore_best_weights=True,
                                verbose=1
                            )

                            callbacks = [lr_scheduler, early_stopping]

                            def inceptionv3_model_build():
                                base_model = InceptionV3(
                                    include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3)
                                )
                                
                                base_model.trainable = False


                                model = models.Sequential([
                                    layers.Input(shape=(224, 224, 3)),
                                    base_model,
                                    layers.GlobalAveragePooling2D(), 
                                    layers.Dense(1024,activation='relu'), 
                                    layers.Dropout(0.3),
                                    layers.Dense(2, activation='softmax')
                                ])

                                optimizer = keras.optimizers.Adam(5e-5)

                                model.compile(
                                    optimizer=optimizer,
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                return model

                            inceptionv3_model = inceptionv3_model_build()
                            inceptionv3_ela_model = inceptionv3_model_build()

                            def xception_model_build():
                                base_model = Xception(
                                    include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3)
                                )
                                
                                base_model.trainable = False


                                model = models.Sequential([
                                    layers.Input(shape=(224, 224, 3)),
                                    base_model,
                                    layers.GlobalAveragePooling2D(), 
                                    layers.Dense(1024,activation='relu'), 
                                    layers.Dropout(0.5),
                                    layers.Dense(2, activation='softmax')
                                ])

                                optimizer = keras.optimizers.Adam(1e-4)

                                model.compile(
                                    optimizer=optimizer,
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                return model

                            xception_model = xception_model_build()
                            xception_ela_model = xception_model_build()
                            
                            def mobilenetv3_model_build():
                                base_model = MobileNetV3Small(
                                    include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3)
                                )
                                
                                base_model.trainable = False


                                model = models.Sequential([
                                    layers.Input(shape=(224, 224, 3)),
                                    base_model,
                                    layers.GlobalAveragePooling2D(), 
                                    layers.Dense(1024,activation='relu'), 
                                    layers.Dropout(0.5),
                                    layers.Dense(2, activation='softmax')
                                ])

                                optimizer = keras.optimizers.Adam(1e-4)

                                model.compile(
                                    optimizer=optimizer,
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                return model

                            mobilenetv3_model = mobilenetv3_model_build()
                            mobilenetv3_ela_model = mobilenetv3_model_build()

                            def vgg16_model_build():
                                base_model = VGG16(
                                    include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3)
                                )
                                
                                base_model.trainable = False


                                model = models.Sequential([
                                    layers.Input(shape=(224, 224, 3)),
                                    base_model,
                                    layers.GlobalAveragePooling2D(), 
                                    layers.Dense(1024,activation='relu'), 
                                    layers.Dropout(0.5),
                                    layers.Dense(2, activation='softmax')
                                ])

                                optimizer = keras.optimizers.Adam(1e-4)

                                model.compile(
                                    optimizer=optimizer,
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                return model

                            vgg16_model = vgg16_model_build()
                            vgg16_ela_model = vgg16_model_build()

                            densenet_ela_history = densenet_ela_model.fit(X_train_ela, Y_train_ela,epochs=10,validation_data=(X_val_ela, Y_val_ela),callbacks=callbacks)

                            densenet_history = densenet_model.fit(X_train, Y_train,epochs=10,validation_data=(X_val, Y_val),callbacks=callbacks)
                
                            inceptionv3_ela_history = inceptionv3_ela_model.fit(X_train_ela, Y_train_ela,epochs=10,validation_data=(X_val_ela, Y_val_ela),callbacks=callbacks)

                            inceptionv3_history = inceptionv3_model.fit(X_train, Y_train,epochs=10,validation_data=(X_val, Y_val),callbacks=callbacks)

                            xception_ela_history = xception_ela_model.fit(X_train_ela, Y_train_ela,epochs=10,validation_data=(X_val_ela, Y_val_ela),callbacks=callbacks)

                            xception_history = xception_model.fit(X_train, Y_train,epochs=10,validation_data=(X_val, Y_val),callbacks=callbacks)
                            
                            mobilenetv3_ela_history = mobilenetv3_ela_model.fit(X_train_ela, Y_train_ela,epochs=10,validation_data=(X_val_ela, Y_val_ela),callbacks=callbacks)

                            mobilenetv3_history = mobilenetv3_model.fit(X_train, Y_train,epochs=10,validation_data=(X_val, Y_val),callbacks=callbacks)

                            vgg16_ela_history = vgg16_ela_model.fit(X_train_ela, Y_train_ela,epochs=10,validation_data=(X_val_ela, Y_val_ela),callbacks=callbacks)

                            vgg16_history = vgg16_model.fit(X_train, Y_train,epochs=10,validation_data=(X_val, Y_val),callbacks=callbacks)

                            """,
                },

                "Evaluation": {
                    "title":"_Evaluation of models:_",
                    "code": """
                            def plot_metrics(history): #plot graph over epochs
                                acc = history.history['accuracy']
                                val_acc = history.history['val_accuracy']
                                loss = history.history['loss']
                                val_loss = history.history['val_loss']
                                epochs = range(1, len(acc) + 1)

                                plt.figure(figsize=(14, 5)) #plot acc
                                plt.subplot(1, 2, 1)
                                plt.plot(epochs, acc, label='Training Accuracy')
                                plt.plot(epochs, val_acc, label='Validation Accuracy')
                                plt.title('Training and Validation Accuracy')
                                plt.xlabel('Epochs')
                                plt.ylabel('Accuracy')
                                plt.legend()

                                plt.subplot(1, 2, 2) # plot loss
                                plt.plot(epochs, loss, label='Training Loss')
                                plt.plot(epochs, val_loss, label='Validation Loss')
                                plt.title('Training and Validation Loss')
                                plt.xlabel('Epochs')
                                plt.ylabel('Loss')
                                plt.legend()

                                plt.tight_layout()
                                plt.show()
                            
                            def evaluate_model(model, X_test, Y_test, class_names=["Real", "Fake"]): #predict test set
                                Y_pred = model.predict(X_test)
                                Y_pred_classes = np.argmax(Y_pred, axis=1)

                                cm = confusion_matrix(Y_test, Y_pred_classes)

                                plt.figure(figsize=(6, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                                plt.ylabel('True Label')
                                plt.xlabel('Predicted Label')
                                plt.title('Confusion Matrix')
                                plt.show()

                                print(classification_report(Y_test, Y_pred_classes, target_names=class_names))
                            
                            plot_metrics(densenet_ela_history)
                            evaluate_model(densenet_ela_model, X_test_ela, Y_test_ela)

                            plot_metrics(densenet_history)
                            evaluate_model(densenet_model, X_test, Y_test)
                
                            plot_metrics(inceptionv3_ela_history)
                            evaluate_model(inceptionv3_ela_model, X_test_ela, Y_test_ela)

                            plot_metrics(inceptionv3_history)
                            evaluate_model(inceptionv3_model, X_test, Y_test)

                            plot_metrics(xception_ela_history)
                            evaluate_model(xception_ela_model, X_test_ela, Y_test_ela)

                            plot_metrics(xception_history)
                            evaluate_model(xception_model, X_test, Y_test)

                            plot_metrics(mobilenetv3_ela_history)
                            evaluate_model(mobilenetv3_ela_model, X_test, Y_test)

                            plot_metrics(mobilenetv3_history)
                            evaluate_model(mobilenetv3_model, X_test, Y_test)

                            plot_metrics(vgg16_ela_history)
                            evaluate_model(vgg16_ela_model, X_test, Y_test)

                            plot_metrics(vgg16_history)
                            evaluate_model(vgg16_model, X_test, Y_test)

                            """,
                },
            }
            
            if code_type in code_snippets:
                st.markdown(f"### {code_snippets[code_type]['title']}")
                st.write("")
                
                st.code(code_snippets[code_type]['code'], language='python')
                st.download_button(
                    label="Download Code",
                    data=code_snippets[code_type]['code'],
                    file_name=f"{code_type.lower()}_code.py",
                    mime="text/plain"
                )
