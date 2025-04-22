# UGP
Data scraping from publically available channel (DEF) for improving Indian Sign Language corpus. Then worked with CISLR-2022 dataset and boosted the accuracy they had 

The repo includes,
1. PPT used for the presentation
2. Final report
3. Multiple colab files and a kaggle file
4. Versatile script to scrape data from the DEF YouTube channel


# Interactive Python Notebook file descriptions
1. Basic_checking_of_feature_sets.ipynb: Just explores the basic I3D and pose modalities and combines the 2 by direct concatenation to see how each fares.
2. One_Shot_Sign_Language_Recognition_Improved_I3D_Features_and_Pose_Velocity_Fusion.ipynb: Improved the I3D feature set by a different pooling, explores different modalities of pose (face, hands, body, face+hands), and different ways of combining both.
3. One_Shot_Sign_Language_Recognition_Improved_Feature_Pooling_and_Fusion_Techniques(PCA,_GeM,_Vel_Attn).ipynb: GeM, PCA pooling on I3D and infusing velocity.
4. Mahalanobis,_PCA_and_Velocity.ipynb: I3D, Mahalanobis, GeM and Velocity and trying to combine these with different methods and hyper-parameters.
5. Experimenting with all different models including combining some of the best models: Different models to combine our best methods and trying new models as well to push top-1, 5 and 10 accuracies.
