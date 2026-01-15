class GroqChatError(Exception):
    pass

class SessionError(GroqChatError):
    pass

class RAGError(GroqChatError):
    pass