const {
    getAllContacts,
    getAllConversations,
    getContact,
    createConversation,
    updateLastMessage,
  } = require("../controllers/conversationController");

  const authenticateToken = require('../middleware/tokenAuthentication');
  
  const router = require("express").Router();
  
  router.post("/createconversation", authenticateToken, createConversation);
  router.post("/updateconversation", authenticateToken, updateLastMessage);
  router.get("/allcontacts/:id", authenticateToken, getAllContacts);
  router.get("/allconversations/:username", authenticateToken, getAllConversations);
  router.get("/contact/:username", authenticateToken, getContact);
  
  module.exports = router;