# RIA Similarity App
The RIA Similarity App is a recommendation system that allows users to select 1) a Registered Investment Adviser (RIA) and 2) a region of the US (or all regions) to be shown up to 20 of the most similar RIAs to the target RIA in the selected region. Access the app (hosted on Heroku free tier so please give it some time for the Dyno to start):

[ria-similarity.herokuapp.com/](https://ria-similarity.herokuapp.com/)

## Model Type
Unsupervised k-Nearest Neighbors

## Training Data
The models used by the application are fit with data obtained from US Securities and Exchange Commission (SEC) Form ADV. The data selected for the models come from Part 1 of the form which captures structured information about each RIA's business, ownership, clients, business practices, and affiliations.

## Intended Use
The application presents similar RIAs to a user's target RIA based on multiple attributes related to business model, scale, and growth. For a user responsible for marketing to RIAs, recruiting advisors from RIAs, or acquiring RIAs; this application could aid in the discovery of firms that are like an RIA the user has had success with in the past — helping the user focus on prospects that might be receptive to the user's value proposition given the similarities to the target RIA.