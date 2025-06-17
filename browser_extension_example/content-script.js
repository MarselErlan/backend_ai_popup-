/**
 * 🎯 Smart Form Filler - Content Script
 * Detects and fills form fields automatically
 */

class FormFiller {
  constructor() {
    this.isEnabled = true;
    this.filledFields = new Set();
    this.init();
  }

  async init() {
    console.log('🎯 Form Filler initialized on:', window.location.href);
    
    // Check if user is logged in
    const userData = await this.getUserData();
    if (!userData.user_id) {
      console.log('👤 User not logged in, form filling disabled');
      return;
    }

    console.log('✅ User logged in:', userData.user_email);
    this.setupEventListeners();
    this.addVisualIndicators();
  }

  async getUserData() {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: 'getStoredData' }, resolve);
    });
  }

  setupEventListeners() {
    // Auto-fill on focus
    document.addEventListener('focusin', (event) => {
      if (this.isFormField(event.target) && !this.filledFields.has(event.target)) {
        this.handleFieldFocus(event.target);
      }
    });

    // Add right-click context menu (future enhancement)
    document.addEventListener('contextmenu', (event) => {
      if (this.isFormField(event.target)) {
        this.lastRightClickedField = event.target;
      }
    });

    // Keyboard shortcut (Ctrl+Shift+F to fill current field)
    document.addEventListener('keydown', (event) => {
      if (event.ctrlKey && event.shiftKey && event.key === 'F') {
        event.preventDefault();
        const activeElement = document.activeElement;
        if (this.isFormField(activeElement)) {
          this.fillField(activeElement);
        }
      }
    });
  }

  isFormField(element) {
    if (!element || !element.tagName) return false;
    
    const tagName = element.tagName.toLowerCase();
    const type = element.type?.toLowerCase();
    
    // Support various input types
    if (tagName === 'input') {
      const supportedTypes = ['text', 'email', 'tel', 'url', 'search', 'password'];
      return supportedTypes.includes(type) || !type;
    }
    
    // Support textareas
    if (tagName === 'textarea') return true;
    
    // Support contenteditable elements
    if (element.contentEditable === 'true') return true;
    
    return false;
  }

  async handleFieldFocus(field) {
    // Add visual indicator
    this.addFieldIndicator(field);
    
    // Small delay to ensure field is ready
    setTimeout(() => {
      this.fillField(field);
    }, 300);
  }

  async fillField(field) {
    if (this.filledFields.has(field)) {
      console.log('🔄 Field already filled, skipping');
      return;
    }

    try {
      // Show loading indicator
      this.showLoadingIndicator(field);
      
      // Get field label/context
      const fieldLabel = this.getFieldLabel(field);
      console.log('🎯 Filling field:', fieldLabel);

      // Request field answer from background script
      const response = await new Promise((resolve) => {
        chrome.runtime.sendMessage({
          action: 'generateFieldAnswer',
          label: fieldLabel,
          url: window.location.href
        }, resolve);
      });

      this.hideLoadingIndicator(field);

      if (response.success !== false && response.answer) {
        // Fill the field
        this.setFieldValue(field, response.answer);
        this.filledFields.add(field);
        this.showSuccessIndicator(field, response.answer);
        
        console.log('✅ Field filled:', fieldLabel, '→', response.answer);
      } else {
        console.log('❌ Failed to get field answer:', response.error);
        this.showErrorIndicator(field);
      }

    } catch (error) {
      console.error('❌ Fill field error:', error);
      this.hideLoadingIndicator(field);
      this.showErrorIndicator(field);
    }
  }

  getFieldLabel(field) {
    // Try multiple methods to get field context
    
    // 1. Placeholder text
    if (field.placeholder && field.placeholder.trim()) {
      return field.placeholder.trim();
    }
    
    // 2. Associated label
    if (field.id) {
      const label = document.querySelector(`label[for="${field.id}"]`);
      if (label && label.textContent.trim()) {
        return label.textContent.trim();
      }
    }
    
    // 3. Parent label
    const parentLabel = field.closest('label');
    if (parentLabel && parentLabel.textContent.trim()) {
      return parentLabel.textContent.trim();
    }
    
    // 4. Aria-label
    if (field.getAttribute('aria-label')) {
      return field.getAttribute('aria-label').trim();
    }
    
    // 5. Name attribute
    if (field.name && field.name.trim()) {
      return field.name.replace(/[_-]/g, ' ').trim();
    }
    
    // 6. Previous text content
    const prevElement = field.previousElementSibling;
    if (prevElement && prevElement.textContent.trim()) {
      return prevElement.textContent.trim();
    }
    
    // 7. Fallback
    return 'Unknown Field';
  }

  setFieldValue(field, value) {
    // Set the value
    field.value = value;
    
    // Trigger events to ensure the form recognizes the change
    const events = ['input', 'change', 'blur'];
    events.forEach(eventType => {
      field.dispatchEvent(new Event(eventType, { bubbles: true }));
    });
    
    // For React/Vue forms
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value'
    ).set;
    nativeInputValueSetter.call(field, value);
    
    field.dispatchEvent(new Event('input', { bubbles: true }));
  }

  // Visual Indicators
  addVisualIndicators() {
    // Add CSS for indicators
    if (!document.getElementById('smart-form-filler-styles')) {
      const style = document.createElement('style');
      style.id = 'smart-form-filler-styles';
      style.textContent = `
        .smart-fill-indicator {
          position: relative;
        }
        .smart-fill-indicator::after {
          content: '🤖';
          position: absolute;
          right: 5px;
          top: 50%;
          transform: translateY(-50%);
          font-size: 12px;
          opacity: 0.7;
          pointer-events: none;
        }
        .smart-fill-loading::after {
          content: '⏳';
          animation: pulse 1s infinite;
        }
        .smart-fill-success::after {
          content: '✅';
          animation: flash 0.5s;
        }
        .smart-fill-error::after {
          content: '❌';
          animation: flash 0.5s;
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.7; }
          50% { opacity: 1; }
        }
        @keyframes flash {
          0%, 100% { opacity: 0.7; }
          50% { opacity: 1; }
        }
      `;
      document.head.appendChild(style);
    }
  }

  addFieldIndicator(field) {
    field.classList.add('smart-fill-indicator');
  }

  showLoadingIndicator(field) {
    field.classList.remove('smart-fill-success', 'smart-fill-error');
    field.classList.add('smart-fill-loading');
  }

  hideLoadingIndicator(field) {
    field.classList.remove('smart-fill-loading');
  }

  showSuccessIndicator(field, answer) {
    field.classList.remove('smart-fill-loading', 'smart-fill-error');
    field.classList.add('smart-fill-success');
    
    // Show tooltip with the answer
    field.title = `Smart Fill: ${answer}`;
    
    // Remove success indicator after 3 seconds
    setTimeout(() => {
      field.classList.remove('smart-fill-success');
    }, 3000);
  }

  showErrorIndicator(field) {
    field.classList.remove('smart-fill-loading', 'smart-fill-success');
    field.classList.add('smart-fill-error');
    
    field.title = 'Smart Fill: Failed to generate answer';
    
    // Remove error indicator after 3 seconds
    setTimeout(() => {
      field.classList.remove('smart-fill-error');
    }, 3000);
  }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new FormFiller();
  });
} else {
  new FormFiller();
}

console.log('🎯 Smart Form Filler content script loaded!'); 