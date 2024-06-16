 const {
    login,
    register,
    logOut,
    getAllContacts,
    getAllConversations,
    getContact,
    createConversation,
    updateLastMessage,
  } = require("../controllers/userController");

  const authenticateToken = require('../middleware/tokenAuthentication');
  
  const router = require("express").Router();
  
  router.post("/login", login);
  router.post("/register", register);
  router.post("/createconversation", authenticateToken, createConversation);
  router.post("/updateconversation", authenticateToken, updateLastMessage);
  router.get("/allcontacts/:id", authenticateToken, getAllContacts);
  router.get("/allconversations/:username", authenticateToken, getAllConversations);
  router.get("/contact/:username", authenticateToken, getContact);
  router.post("/logout/:id", authenticateToken, logOut);
  
  module.exports = router;
  