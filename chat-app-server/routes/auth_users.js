const {
    login,
    register,
    getAllContacts,
    getAllConversations,
    getContact,
    // setAvatar,
    // logOut,
  } = require("../controllers/userController");
  
  const router = require("express").Router();
  
  router.post("/login", login);
  router.post("/register", register);
  router.get("/allcontacts/:id", getAllContacts);
  router.get("/allconversations/:username", getAllConversations);
  router.get("/contact/:username", getContact);
//   router.post("/setavatar/:id", setAvatar);
//   router.get("/logout/:id", logOut);
  
  module.exports = router;
  