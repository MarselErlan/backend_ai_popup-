# 🚀 Simplified Authentication System for Browser Extensions

## Overview

The backend now supports **simplified authentication** where the browser extension only needs to send:

1. **Question/Field Label**
2. **User ID**

No JWT tokens, no complex headers - just simple user_id validation in the backend!

## 🎯 Main Endpoint (Simplified Auth)

### `POST /api/generate-field-answer`

```javascript
// Browser Extension Usage
const response = await fetch(
  "http://localhost:8000/api/generate-field-answer",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      label: "Full Name", // Field question/label
      url: "https://example.com", // Current page URL
      user_id: "your-user-id-here", // User ID from registration
    }),
  }
);

const result = await response.json();
console.log(result.answer); // "John Doe"
```

**Response:**

```json
{
  "answer": "John Doe",
  "data_source": "resume_vectordb",
  "reasoning": "Early exit optimization - 95.0% satisfaction from resume data",
  "status": "success",
  "performance_metrics": {
    "processing_time_seconds": 1.2,
    "backend_authorization": true
  }
}
```

## 📝 User Registration (One-Time Setup)

### `POST /api/simple/register`

```javascript
// Register user and get user_id (store this in extension storage)
const response = await fetch("http://localhost:8000/api/simple/register", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    email: "user@example.com",
    password: "password123",
  }),
});

const result = await response.json();
// Store result.user_id in browser extension storage
localStorage.setItem("user_id", result.user_id);
```

**Response:**

```json
{
  "status": "registered",
  "user_id": "c1f38b80-af47-4027-8c4e-7b2cd7a90d24",
  "email": "user@example.com",
  "message": "User registered successfully - save this user_id for future requests"
}
```

### `POST /api/simple/login`

```javascript
// Login existing user and get user_id
const response = await fetch("http://localhost:8000/api/simple/login", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    email: "user@example.com",
    password: "password123",
  }),
});

const result = await response.json();
localStorage.setItem("user_id", result.user_id);
```

## 🔍 User Validation

### `GET /api/validate-user/{user_id}`

```javascript
// Validate user_id before making requests
const userId = localStorage.getItem("user_id");
const response = await fetch(
  `http://localhost:8000/api/validate-user/${userId}`
);
const result = await response.json();

if (result.status === "valid") {
  // User is valid, can make field answer requests
  console.log(`User type: ${result.user_type}`); // "registered" or "demo"
}
```

## 🎯 Demo Mode (No Registration Required)

### For Testing: Use `user_id: "default"`

```javascript
// No registration needed for demo
const response = await fetch(
  "http://localhost:8000/api/generate-field-answer",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      label: "Email Address",
      url: "https://example.com",
      user_id: "default", // Demo user
    }),
  }
);
```

### Or use dedicated demo endpoint:

```javascript
const response = await fetch(
  "http://localhost:8000/api/demo/generate-field-answer",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      label: "Email Address",
      url: "https://example.com",
      // No user_id needed for demo endpoint
    }),
  }
);
```

## 🏗️ Browser Extension Implementation

### 1. Extension Storage Setup

```javascript
// content-script.js or background.js
class AuthManager {
  static async getUserId() {
    return new Promise((resolve) => {
      chrome.storage.local.get(["user_id"], (result) => {
        resolve(result.user_id || "default");
      });
    });
  }

  static async setUserId(userId) {
    return new Promise((resolve) => {
      chrome.storage.local.set({ user_id: userId }, resolve);
    });
  }

  static async registerUser(email, password) {
    const response = await fetch("http://localhost:8000/api/simple/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const result = await response.json();
    if (result.status === "registered" || result.status === "exists") {
      await this.setUserId(result.user_id);
      return result;
    }
    throw new Error(result.message);
  }
}
```

### 2. Field Answer Integration

```javascript
// content-script.js
class FormFiller {
  static async getFieldAnswer(fieldLabel, pageUrl) {
    const userId = await AuthManager.getUserId();

    const response = await fetch(
      "http://localhost:8000/api/generate-field-answer",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label: fieldLabel,
          url: pageUrl,
          user_id: userId,
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    return result.answer;
  }

  static async fillFormField(inputElement) {
    const fieldLabel = this.getFieldLabel(inputElement);
    const pageUrl = window.location.href;

    try {
      const answer = await this.getFieldAnswer(fieldLabel, pageUrl);
      if (answer) {
        inputElement.value = answer;
        inputElement.dispatchEvent(new Event("input", { bubbles: true }));
      }
    } catch (error) {
      console.error("Field filling failed:", error);
    }
  }

  static getFieldLabel(inputElement) {
    // Extract field label logic here
    return (
      inputElement.placeholder ||
      inputElement.name ||
      inputElement.getAttribute("aria-label") ||
      "Unknown Field"
    );
  }
}

// Auto-fill on focus
document.addEventListener("focusin", async (event) => {
  if (event.target.tagName === "INPUT" && event.target.type === "text") {
    await FormFiller.fillFormField(event.target);
  }
});
```

## 🔄 Available Authentication Options

1. **Simplified Auth** (Recommended for Extensions):

   - `POST /api/generate-field-answer` - Just send user_id
   - `POST /api/simple/register` - Get user_id
   - `POST /api/simple/login` - Get user_id
   - `GET /api/validate-user/{user_id}` - Validate user_id

2. **Demo Mode** (No Registration):

   - `POST /api/demo/generate-field-answer` - No user_id needed
   - Use `user_id: "default"` in main endpoint

3. **Full JWT Auth** (Advanced Users):
   - `POST /api/auth/register` - Returns JWT token
   - `POST /api/auth/login` - Returns JWT token
   - `POST /api/auth/generate-field-answer` - Requires Authorization header

## ✅ Key Benefits

- **Simple**: Extension only needs to store/send user_id
- **No Tokens**: No JWT token management needed
- **Backend Validation**: All security handled server-side
- **Demo Ready**: Works with default user for testing
- **Performance**: Optimized with caching and early exit
- **Multiple Options**: Choose the auth level you need

Perfect for browser extensions where managing JWT tokens would be complex! 🎉
