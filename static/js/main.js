document.getElementById('chat-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission
    sendMessage();
    });

document.getElementById('user-input').addEventListener('keydown', function(event) {
    // Check if the pressed key is Enter
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent the default behavior (e.g., line break in the input)
        sendMessage();
        }
    });

async function sendMessage() {
    try {
        var userMessage = document.getElementById('user-input').value;
        document.getElementById('user-input').value = '';

        // Send user message to the server for a response
        const response = await fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'user_message=' + encodeURIComponent(userMessage),
        });

        const data = await response.text();

        // Display user message immediately
        var userMessageElement = '<div class ="user-message-container"><div style="padding:5px;">Me</div><div class="chat-message user-message">' + userMessage + '</div></div>';
        document.getElementById('chat-display').innerHTML += userMessageElement;

        // Introduce a delay before displaying the chatbot's response
        setTimeout(function () {
            // Display chatbot response in the chat display
            var chatbotMessageElement = '<div class ="chatbot-message-container"><div style="padding:5px;">WeatherBot</div><div class="chat-message chatbot-message">' + data + '</div></div>';
            document.getElementById('chat-display').innerHTML += chatbotMessageElement;

            // Scroll the chat display to the bottom
            scrollChatToBottom();
        }, 500); // Adjust the delay duration (in milliseconds) as needed

    } catch (error) {
        console.error('Error:', error);
    }
}


function scrollChatToBottom() {
    var chatDisplay = document.getElementById('chat-display');
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
}