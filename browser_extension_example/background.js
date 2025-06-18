// API Configuration
const API_BASE_URL = 'http://localhost:8000';

/**
 * 🔑 Session Management Class
 * Handles user authentication and session management
 */
class SessionManager {
  static async getStoredData() {
    return new Promise((resolve) => {
      chrome.storage.local.get(['session_id', 'user_id', 'user_email'], resolve);
    });
  }

  static async setStoredData(data) {
    return new Promise((resolve) => {
      chrome.storage.local.set(data, resolve);
    });
  }

  static async clearStoredData() {
    return new Promise((resolve) => {
      chrome.storage.local.clear(resolve);
    });
  }

  /**
   * 📝 Register new user and create session
   */
  static async registerUser(email, password) {
    try {
      console.log('🔐 Registering user:', email);
      
      // Step 1: Register user
      const registerResponse = await fetch(`${API_BASE_URL}/api/simple/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (!registerResponse.ok) {
        throw new Error(`Registration failed: ${registerResponse.status}`);
      }

      const userData = await registerResponse.json();
      console.log('✅ User registered:', userData);

      // Step 2: Create session
      const sessionData = await this.createSession(userData.user_id);
      
      // Step 3: Store data permanently
      await this.setStoredData({
        session_id: sessionData.session_id,
        user_id: userData.user_id,
        user_email: userData.email
      });

      return {
        success: true,
        user: userData,
        session: sessionData
      };

    } catch (error) {
      console.error('❌ Registration failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 🔑 Create session for existing user
   */
  static async createSession(userId) {
    const deviceInfo = `Chrome Extension v1.0 - ${navigator.userAgent.split(' ')[0]}`;
    
    const response = await fetch(`${API_BASE_URL}/api/session/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        device_info: deviceInfo
      })
    });

    if (!response.ok) {
      throw new Error(`Session creation failed: ${response.status}`);
    }

    return await response.json();
  }

  /**
   * 👤 Get current user info
   */
  static async getCurrentUser() {
    try {
      const { session_id } = await this.getStoredData();
      
      if (!session_id) {
        return { success: false, error: 'No session found' };
      }

      const response = await fetch(`${API_BASE_URL}/api/session/current/${session_id}`);
      
      if (!response.ok) {
        if (response.status === 404) {
          // Session expired, clear stored data
          await this.clearStoredData();
          return { success: false, error: 'Session expired' };
        }
        throw new Error(`Failed to get current user: ${response.status}`);
      }

      const user = await response.json();
      return { success: true, user };

    } catch (error) {
      console.error('❌ Get current user failed:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * 🚪 Logout user (deactivate session)
   */
  static async logout() {
    try {
      const { session_id } = await this.getStoredData();
      
      if (session_id) {
        await fetch(`${API_BASE_URL}/api/session/${session_id}`, {
          method: 'DELETE'
        });
      }
      
      await this.clearStoredData();
      return { success: true };

    } catch (error) {
      console.error('❌ Logout failed:', error);
      // Still clear local data even if API call fails
      await this.clearStoredData();
      return { success: false, error: error.message };
    }
  }

  /**
   * 🎯 Generate field answer - FIXED to use demo endpoint
   */
  static async generateFieldAnswer(label, url) {
    try {
      console.log('🎯 Generating field answer for:', label);
      
      // ✅ FIXED: Use demo endpoint (no authentication required)
      const response = await fetch(`${API_BASE_URL}/api/demo/generate-field-answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label,
          url,
          user_id: 'default'  // Demo user
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Field answer failed: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      console.log('✅ Field answer generated:', result);
      return result;

    } catch (error) {
      console.error('❌ Generate field answer failed:', error);
      throw error;
    }
  }
}

// Message handling from popup and content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('📨 Background received message:', request);

  switch (request.action) {
    case 'register':
      SessionManager.registerUser(request.email, request.password)
        .then(sendResponse);
      return true; // Keep message channel open for async response

    case 'getCurrentUser':
      SessionManager.getCurrentUser()
        .then(sendResponse);
      return true;

    case 'logout':
      SessionManager.logout()
        .then(sendResponse);
      return true;

    case 'generateFieldAnswer':
      SessionManager.generateFieldAnswer(request.label, request.url)
        .then(sendResponse)
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;

    case 'getStoredData':
      SessionManager.getStoredData()
        .then(sendResponse);
      return true;

    default:
      console.warn('❓ Unknown action:', request.action);
      sendResponse({ success: false, error: 'Unknown action' });
  }
});

// Extension installation
chrome.runtime.onInstalled.addListener(() => {
  console.log('🚀 Smart Form Filler Extension installed!');
});

console.log('🔑 Background service worker loaded!'); 