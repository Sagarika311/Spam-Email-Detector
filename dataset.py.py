import pandas as pd

# Expanded sample data for demonstration
data = {
    'email': [
        # Spam emails
        'Congratulations! You have won a $1,000 Walmart gift card. Click here to claim your prize.',
        'Dear user, your account has been compromised. Please reset your password immediately.',
        'Get paid to work from home! Sign up now for a free trial.',
        'Limited time offer: Buy one get one free on all products!',
        'Urgent: Your invoice is attached. Please review it immediately.',
        'You have been selected for a chance to get a free iPhone!',
        'Win a vacation to Bahamas! Click to enter now.',
        'Exclusive deal just for you! Get 90% off on your next purchase.',
        'Your account has been suspended. Click here to restore access.',
        'Claim your free gift card now! Supplies are limited.',

        # Valid emails
        'Hi there! I hope you are having a great day. Just wanted to check in.',
        'Reminder: Your appointment is scheduled for tomorrow at 3 PM.',
        'You have received a new message from your friend. Check it out!',
        'Meeting rescheduled to next week. Please confirm your availability.',
        'Your subscription has been renewed successfully. Thank you for being a valued customer.',
        'Thank you for your order! Your package will arrive shortly.',
        'Can we schedule a call to discuss the project updates?',
        'Here is the report you requested. Let me know if you need any changes.',
        'The team is doing great work. Keep it up!',
        'Looking forward to our meeting next week. Have a great day!'
    ],
    'label': [
        # Labels for spam emails
        'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam',
        
        # Labels for valid emails
        'valid', 'valid', 'valid', 'valid', 'valid', 'valid', 'valid', 'valid', 'valid', 'valid'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('spam_dataset.csv', index=False)

print("CSV file 'spam_dataset.csv' created successfully!")