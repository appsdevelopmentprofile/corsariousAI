---
title: OCR
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---



# Proposed architecture

Modules & Their Setup Needs
Outlook Listener (Graph API)

External Library/Service: You’ll need to use the Microsoft Graph API to connect to Outlook and retrieve emails.
API Keys: Set up an application in the Microsoft Azure portal to obtain client ID, client secret, and tenant ID for accessing the Microsoft Graph API.
Railway Dashboard: Add the client ID, client secret, and tenant ID as environment variables in the Railway dashboard (e.g., OUTLOOK_CLIENT_ID, OUTLOOK_CLIENT_SECRET, OUTLOOK_TENANT_ID).
Code: Your GitHub repository should include the code to authenticate and fetch emails using these credentials.
OCR Processor (Tesseract)

External Library: Install Tesseract OCR via your requirements.txt or include it in the Dockerfile.
Railway Configuration: No additional Railway configuration is needed, as Tesseract will run directly in the Docker container.
Code: Include the OCR processing code in your GitHub repository.
Generative AI (OpenAI, GPT)

External Service: Sign up for the OpenAI API or a similar generative AI service.
API Key: Obtain the API key from the OpenAI dashboard.
Railway Dashboard: Add the OpenAI API key as an environment variable (e.g., OPENAI_API_KEY) in the Railway dashboard.
Code: Add code in your repository to interact with the OpenAI API using the API key for generating reports based on OCR-extracted data.
Cloud Storage Integration (AWS S3/Google Cloud/OneDrive)

External Service: You’ll need access credentials to the customer’s cloud storage platform.
For AWS S3: Create or obtain AWS access keys (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY).
For Google Cloud Storage: Use a JSON service account key.
For OneDrive: Similar to the Outlook Listener, use the Microsoft Graph API and the corresponding client credentials.
Railway Dashboard: Store credentials in Railway environment variables (e.g., AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, GOOGLE_CLOUD_KEY, ONEDRIVE_CLIENT_ID).
Code: Include code in the repository to upload files to the specified cloud storage provider.
Email Sender (SMTP/SendGrid)

External Service: Use SMTP for standard email or a service like SendGrid if you want more features or reliability.
API Key: For SendGrid, sign up and get an API key. For SMTP, use SMTP credentials (like those from a Gmail or Outlook account) to send emails.
Railway Dashboard: Store email credentials or SendGrid API key in Railway environment variables (e.g., SENDGRID_API_KEY, SMTP_USER, SMTP_PASSWORD, etc.).
Code: Include email-sending code in your repository, making use of environment variables for sensitive credentials.
Web Dashboard (Streamlit or FastAPI)

Code: Include all necessary code in your GitHub repository to launch a Streamlit or FastAPI application.
Railway Configuration: No special Railway configuration beyond the Docker setup is needed, though port configuration may be required. Railway typically uses port 80, so make sure Streamlit/FastAPI listens to that port or configures it based on an environment variable (e.g., PORT).
