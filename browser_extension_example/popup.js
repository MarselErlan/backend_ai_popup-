/**
 * 🔑 Smart Form Filler - Popup Interface
 * Handles user registration, login, and extension management
 */

class PopupManager {
  constructor() {
    this.init();
  }

  async init() {
    console.log('🔑 Popup manager initialized');
    
    // Show loading initially
    this.showLoading();
    
    // Check current user status
    await this.checkUserStatus();
    
    // Setup event listeners
    this.setupEventListeners();
  }

  async checkUserStatus() {
    try {
      // Get current user from background script
      const response = await this.sendMessage({ action: 'getCurrentUser' });
      
      if (response.success && response.user) {
        this.showLoggedInState(response.user);
      } else {
        this.showLoggedOutState();
      }
    } catch (error) {
      console.error('❌ Check user status failed:', error);
      this.showLoggedOutState();
    }
  }

  setupEventListeners() {
    // Register form submission
    document.getElementById('register-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleRegister();
    });

    // Logout button
    document.getElementById('logout-btn').addEventListener('click', () => {
      this.handleLogout();
    });

    // Upload documents button
    document.getElementById('upload-docs-btn').addEventListener('click', () => {
      this.openUploadPage();
    });

    // Test form fill button
    document.getElementById('test-fill-btn').addEventListener('click', () => {
      this.testFormFill();
    });
  }

  async handleRegister() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    if (!email || !password) {
      this.showError('Please enter both email and password');
      return;
    }

    try {
      this.showLoading();
      this.hideMessages();

      // Register user via background script
      const response = await this.sendMessage({
        action: 'register',
        email,
        password
      });

      if (response.success) {
        this.showSuccess('Registration successful! You are now logged in.');
        
        // Clear form
        document.getElementById('email').value = '';
        document.getElementById('password').value = '';
        
        // Show logged in state
        setTimeout(() => {
          this.checkUserStatus();
        }, 1500);

      } else {
        this.showError(response.error || 'Registration failed');
        this.showLoggedOutState();
      }

    } catch (error) {
      console.error('❌ Registration error:', error);
      this.showError('Registration failed: ' + error.message);
      this.showLoggedOutState();
    }
  }

  async handleLogout() {
    try {
      this.showLoading();
      
      const response = await this.sendMessage({ action: 'logout' });
      
      if (response.success) {
        this.showSuccess('Logged out successfully');
        setTimeout(() => {
          this.showLoggedOutState();
        }, 1000);
      } else {
        this.showError('Logout failed: ' + response.error);
      }

    } catch (error) {
      console.error('❌ Logout error:', error);
      this.showError('Logout failed: ' + error.message);
    }
  }

  openUploadPage() {
    // Open the API documentation/upload page
    chrome.tabs.create({
      url: 'http://localhost:8000/docs'
    });
  }

  async testFormFill() {
    try {
      // Get current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      // Inject a test form and fill it
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: this.createTestForm
      });

      this.showSuccess('Test form created! Try focusing on the fields.');

    } catch (error) {
      console.error('❌ Test form fill error:', error);
      this.showError('Test failed: ' + error.message);
    }
  }

  // This function will be injected into the page
  createTestForm() {
    // Remove existing test form
    const existingForm = document.getElementById('smart-fill-test-form');
    if (existingForm) {
      existingForm.remove();
    }

    // Create test form
    const testForm = document.createElement('div');
    testForm.id = 'smart-fill-test-form';
    testForm.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      width: 300px;
      background: white;
      border: 2px solid #667eea;
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      z-index: 10000;
      font-family: Arial, sans-serif;
    `;

    testForm.innerHTML = `
      <h3 style="margin: 0 0 15px 0; color: #333;">🧪 Smart Fill Test Form</h3>
      <div style="margin-bottom: 10px;">
        <label style="display: block; margin-bottom: 5px; color: #666;">Full Name:</label>
        <input type="text" placeholder="Your full name" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box;">
      </div>
      <div style="margin-bottom: 10px;">
        <label style="display: block; margin-bottom: 5px; color: #666;">Email Address:</label>
        <input type="email" placeholder="your@email.com" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box;">
      </div>
      <div style="margin-bottom: 10px;">
        <label style="display: block; margin-bottom: 5px; color: #666;">Phone Number:</label>
        <input type="tel" placeholder="(555) 123-4567" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box;">
      </div>
      <button onclick="this.parentElement.remove()" style="background: #f44336; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; float: right;">Close</button>
      <div style="clear: both;"></div>
      <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">💡 Focus on any field to test auto-fill!</p>
    `;

    document.body.appendChild(testForm);
  }

  // UI State Management
  showLoading() {
    document.getElementById('loading').classList.add('show');
    document.getElementById('logged-in-state').classList.add('hidden');
    document.getElementById('logged-out-state').classList.add('hidden');
  }

  showLoggedInState(user) {
    document.getElementById('loading').classList.remove('show');
    document.getElementById('logged-out-state').classList.add('hidden');
    document.getElementById('logged-in-state').classList.remove('hidden');

    // Update user info
    document.getElementById('user-email').textContent = user.email;
    document.getElementById('session-info').textContent = `Session: ${user.session_id.substring(0, 8)}...`;
  }

  showLoggedOutState() {
    document.getElementById('loading').classList.remove('show');
    document.getElementById('logged-in-state').classList.add('hidden');
    document.getElementById('logged-out-state').classList.remove('hidden');
  }

  showError(message) {
    const errorEl = document.getElementById('error-message');
    errorEl.textContent = message;
    errorEl.classList.remove('hidden');
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
      errorEl.classList.add('hidden');
    }, 5000);
  }

  showSuccess(message) {
    const successEl = document.getElementById('success-message');
    successEl.textContent = message;
    successEl.classList.remove('hidden');
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
      successEl.classList.add('hidden');
    }, 3000);
  }

  hideMessages() {
    document.getElementById('error-message').classList.add('hidden');
    document.getElementById('success-message').classList.add('hidden');
  }

  // Helper method to send messages to background script
  sendMessage(message) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(message, resolve);
    });
  }
}

// Initialize popup when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new PopupManager();
});

console.log('🔑 Popup script loaded!'); 