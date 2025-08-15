"""User authentication and session management for the dashboard."""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional

from ..config.settings import Settings
from .. import settings

logger = logging.getLogger(__name__)


class AuthManager:
    """Simple authentication manager for dashboard access."""
    
    def __init__(self, settings_obj: Settings = None):
        """Initialize the authentication manager."""
        self.settings = settings_obj or settings
        self._sessions: Dict[str, Dict] = {}
        self._users = self._load_users()
        
    def _load_users(self) -> Dict[str, str]:
        """Load user credentials. In production, this would come from a database."""
        # Simple hardcoded users for demo - in production use proper user management
        users = {
            "admin": self._hash_password("admin123"),
            "viewer": self._hash_password("viewer123"),
            "demo": self._hash_password("demo123")
        }
        return users
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return a session token.
        
        Args:
            username: The username
            password: The password
            
        Returns:
            Session token if authentication successful, None otherwise
        """
        if username not in self._users:
            logger.warning(f"Authentication failed: unknown user {username}")
            return None
            
        hashed_password = self._hash_password(password)
        if self._users[username] != hashed_password:
            logger.warning(f"Authentication failed: incorrect password for {username}")
            return None
        
        # Generate session token
        session_token = secrets.token_urlsafe(32)
        
        # Store session
        self._sessions[session_token] = {
            "username": username,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "role": self._get_user_role(username)
        }
        
        logger.info(f"User {username} authenticated successfully")
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate a session token.
        
        Args:
            session_token: The session token to validate
            
        Returns:
            Session data if valid, None otherwise
        """
        if not session_token or session_token not in self._sessions:
            return None
        
        session = self._sessions[session_token]
        
        # Check if session has expired (24 hours)
        if datetime.utcnow() - session["created_at"] > timedelta(hours=24):
            self.logout_user(session_token)
            return None
        
        # Check for inactivity (2 hours)
        if datetime.utcnow() - session["last_activity"] > timedelta(hours=2):
            self.logout_user(session_token)
            return None
        
        # Update last activity
        session["last_activity"] = datetime.utcnow()
        
        return session
    
    def logout_user(self, session_token: str) -> bool:
        """Logout a user by removing their session.
        
        Args:
            session_token: The session token
            
        Returns:
            True if logout successful, False otherwise
        """
        if session_token in self._sessions:
            username = self._sessions[session_token]["username"]
            del self._sessions[session_token]
            logger.info(f"User {username} logged out")
            return True
        return False
    
    def _get_user_role(self, username: str) -> str:
        """Get the role for a user."""
        role_mapping = {
            "admin": "admin",
            "viewer": "viewer", 
            "demo": "viewer"
        }
        return role_mapping.get(username, "viewer")
    
    def has_permission(self, session_token: str, permission: str) -> bool:
        """Check if a user has a specific permission.
        
        Args:
            session_token: The session token
            permission: The permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        session = self.validate_session(session_token)
        if not session:
            return False
        
        role = session.get("role", "viewer")
        
        # Define permissions
        permissions = {
            "admin": ["view", "export", "admin"],
            "viewer": ["view", "export"]
        }
        
        user_permissions = permissions.get(role, [])
        return permission in user_permissions
    
    def get_active_sessions(self) -> Dict[str, Dict]:
        """Get all active sessions (admin only)."""
        # Clean up expired sessions first
        expired_tokens = []
        for token, session in self._sessions.items():
            if datetime.utcnow() - session["created_at"] > timedelta(hours=24):
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self._sessions[token]
        
        return {
            token: {
                "username": session["username"],
                "role": session["role"],
                "created_at": session["created_at"].isoformat(),
                "last_activity": session["last_activity"].isoformat()
            }
            for token, session in self._sessions.items()
        }
    
    def create_login_form(self) -> str:
        """Create a simple HTML login form."""
        return """
        <div style="max-width: 400px; margin: 100px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
            <h2 style="text-align: center; margin-bottom: 20px;">Weather Dashboard Login</h2>
            <form id="login-form">
                <div style="margin-bottom: 15px;">
                    <label for="username" style="display: block; margin-bottom: 5px;">Username:</label>
                    <input type="text" id="username" name="username" required 
                           style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                </div>
                <div style="margin-bottom: 20px;">
                    <label for="password" style="display: block; margin-bottom: 5px;">Password:</label>
                    <input type="password" id="password" name="password" required
                           style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                </div>
                <button type="submit" 
                        style="width: 100%; padding: 10px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Login
                </button>
            </form>
            <div style="margin-top: 20px; font-size: 12px; color: #666;">
                <p>Demo accounts:</p>
                <ul>
                    <li>admin / admin123 (full access)</li>
                    <li>viewer / viewer123 (view only)</li>
                    <li>demo / demo123 (view only)</li>
                </ul>
            </div>
        </div>
        """
