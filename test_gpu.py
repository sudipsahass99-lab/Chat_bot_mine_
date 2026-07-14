class IITKGPChatbot {
    constructor() {
        this.conversationId = null;
        this.currentConversationTitle = 'New Conversation';
        this.initializeEventListeners();
        this.loadConversations();
        this.startNewConversation();
    }

    initializeEventListeners() {
        const sendButton = document.getElementById('sendButton');
        const messageInput = document.getElementById('messageInput');
        const newChatBtn = document.getElementById('newChatBtn');

        sendButton.addEventListener('click', () => this.sendMessage());
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        newChatBtn.addEventListener('click', () => this.startNewConversation());
    }

    async loadConversations() {
        try {
            const response = await fetch('/api/conversations');
            const data = await response.json();
            
            if (data.conversations) {
                this.renderConversations(data.conversations);
            }
        } catch (error) {
            console.error('Error loading conversations:', error);
        }
    }

    renderConversations(conversations) {
        const conversationsList = document.getElementById('conversationsList');
        
        if (conversations.length === 0) {
            conversationsList.innerHTML = `
                <div class="empty-conversations">
                    <i class="fas fa-comments"></i>
                    <p>No conversations yet</p>
                </div>
            `;
            return;
        }

        conversationsList.innerHTML = conversations.map(conv => `
            <div class="conversation-item ${conv.id === this.conversationId ? 'active' : ''}" 
                 data-conversation-id="${conv.id}">
                <div class="conversation-title">${this.escapeHtml(conv.title)}</div>
                <div class="conversation-meta">
                    ${this.formatDate(conv.updated_at)} • ${conv.message_count} messages
                </div>
                <div class="conversation-actions">
                    <button class="btn btn-sm btn-outline-primary load-conversation" 
                            data-conversation-id="${conv.id}">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger delete-conversation ms-1" 
                            data-conversation-id="${conv.id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');

        // Add event listeners
        document.querySelectorAll('.load-conversation').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const convId = e.target.closest('button').dataset.conversationId;
                this.loadConversation(convId);
            });
        });

        document.querySelectorAll('.delete-conversation').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const convId = e.target.closest('button').dataset.conversationId;
                this.deleteConversation(convId);
            });
        });

        document.querySelectorAll('.conversation-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (!e.target.closest('button')) {
                    const convId = item.dataset.conversationId;
                    this.loadConversation(convId);
                }
            });
        });
    }

    async loadConversation(conversationId) {
        try {
            const response = await fetch(`/api/conversation/${conversationId}`);
            const data = await response.json();
            
            if (data.error) {
                alert('Error loading conversation: ' + data.error);
                return;
            }

            this.conversationId = conversationId;
            this.currentConversationTitle = data.title;
            this.updateConversationTitle();
            this.renderMessages(data.messages);
            this.loadConversations(); // Refresh sidebar
        } catch (error) {
            console.error('Error loading conversation:', error);
            alert('Failed to load conversation');
        }
    }

    async deleteConversation(conversationId) {
        if (!confirm('Are you sure you want to delete this conversation?')) {
            return;
        }

        try {
            const response = await fetch(`/api/conversation/${conversationId}`, {
                method: 'DELETE'
            });
            const data = await response.json();
            
            if (data.success) {
                if (this.conversationId === conversationId) {
                    this.startNewConversation();
                }
                this.loadConversations();
            } else {
                alert('Failed to delete conversation');
            }
        } catch (error) {
            console.error('Error deleting conversation:', error);
            alert('Failed to delete conversation');
        }
    }

    renderMessages(messages) {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '';

        // Filter out system messages and only show user/assistant messages
        const userMessages = messages.filter(msg => 
            msg.role === 'user' || msg.role === 'assistant'
        );

        if (userMessages.length === 0) {
            this.addWelcomeMessage();
            return;
        }

        userMessages.forEach(msg => {
            // Use the correct class for bot messages (assistant role)
            const role = msg.role === 'assistant' ? 'bot' : msg.role;
            this.addMessage(msg.content, role, null, true); // true means it's from loaded conversation
        });

        // Scroll to bottom after rendering
        setTimeout(() => {
            const chatMessagesContainer = document.querySelector('.chat-messages-container');
            if (chatMessagesContainer) {
                chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
            }
        }, 100);
    }

    async startNewConversation() {
        try {
            const response = await fetch('/api/new_conversation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            this.conversationId = data.conversation_id;
            this.currentConversationTitle = 'New Conversation';
            this.updateConversationTitle();
            this.clearChat();
            this.addWelcomeMessage();
            this.loadConversations();
        } catch (error) {
            console.error('Error starting new conversation:', error);
        }
    }

    updateConversationTitle() {
        const titleElement = document.getElementById('currentConversationTitle');
        if (titleElement) {
            titleElement.textContent = this.currentConversationTitle;
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message) return;

        // Add user message to chat
        this.addMessage(message, 'user');
        messageInput.value = '';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.conversationId
                })
            });

            const data = await response.json();

            // Remove typing indicator
            this.removeTypingIndicator();

            if (data.error) {
                this.addMessage(`Sorry, I encountered an error: ${data.error}`, 'bot');
            } else {
                this.addMessage(data.response, 'bot', data.response_time);
                this.conversationId = data.conversation_id;
                this.loadConversations(); // Refresh sidebar to update title
            }
        } catch (error) {
            this.removeTypingIndicator();
            this.addMessage('Sorry, I encountered a network error. Please try again.', 'bot');
            console.error('Error sending message:', error);
        }
    }

    addMessage(text, sender, responseTime = null, fromLoad = false) {
        const chatMessages = document.getElementById('chatMessages');
        
        // Remove welcome message if it exists and we're adding a real message
        if (!fromLoad) {
            const welcomeMessage = chatMessages.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        // Format the text with basic markdown-like formatting
        let formattedText = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/\n/g, '<br>') // Line breaks
            .replace(/\d+\.\s/g, '<br>$&') // Numbered lists
            .replace(/•\s/g, '<br>• '); // Bullet points
        
        bubbleDiv.innerHTML = formattedText;

        messageDiv.appendChild(bubbleDiv);

        if (responseTime) {
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = `Response time: ${responseTime}`;
            messageDiv.appendChild(timeDiv);
        }

        chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        setTimeout(() => {
            const chatMessagesContainer = document.querySelector('.chat-messages-container');
            if (chatMessagesContainer) {
                chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
            }
        }, 50);
    }

    addWelcomeMessage() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = `
            <div class="welcome-message text-center">
                <i class="fas fa-robot fa-3x kgp-blue mb-3"></i>
                <h5>Hello! I'm your IIT KGP Senior</h5>
                <p class="text-muted">I can help you with information about academics, campus life, admissions, and more. What would you like to know?</p>
            </div>
        `;
    }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '';
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chatMessages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot';
        typingDiv.id = 'typingIndicator';
        
        typingDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        chatMessages.appendChild(typingDiv);
        
        // Scroll to bottom to show typing indicator
        setTimeout(() => {
            const chatMessagesContainer = document.querySelector('.chat-messages-container');
            if (chatMessagesContainer) {
                chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
            }
        }, 50);
    }

    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays === 1) {
            return 'Today';
        } else if (diffDays === 2) {
            return 'Yesterday';
        } else if (diffDays <= 7) {
            return `${diffDays - 1} days ago`;
        } else {
            return date.toLocaleDateString();
        }
    }
}

// Initialize chatbot when page loads
document.addEventListener('DOMContentLoaded', function() {
    new IITKGPChatbot();
});