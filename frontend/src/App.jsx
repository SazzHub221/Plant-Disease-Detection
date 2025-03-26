import { useState, useEffect, useCallback } from "react";
import { styled } from "@mui/material/styles";
import {
  AppBar,
  Toolbar,
  Typography,
  Avatar,
  Container,
  Card,
  CardContent,
  Paper,
  CardActionArea,
  CardMedia,
  Grid,
  TableContainer,
  Table,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
  Button,
  CircularProgress,
} from "@mui/material";
import { useDropzone } from "react-dropzone";
import ClearIcon from "@mui/icons-material/Clear";
import cblogo from "./assets/cblogo.png";
import bgImage from "./assets/bg.png";
import axios from "axios";

const ColorButton = styled(Button)(({ theme }) => ({
  color: theme.palette.common.white,
  backgroundColor: theme.palette.common.white,
  "&:hover": {
    backgroundColor: "rgba(255, 255, 255, 0.8)",
  },
}));

const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [image, setImage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  let confidence = 0;

  const sendFile = useCallback(async () => {
    if (image) {
      let formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("model_path", "saved_models/Keras1.keras");
      
      try {
        setIsLoading(true);
        let res = await axios({
          method: "post",
          url: import.meta.env.VITE_API_URL || "https://plant-disease-detection-yz5v.onrender.com/predict",
          data: formData,
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        if (res.status === 200) {
          setData(res.data);
        }
      } catch (error) {
        console.error("Error sending file:", error);
        // Optionally set some error state here
      } finally {
        setIsLoading(false);
      }
    }
  }, [image, selectedFile]);

  const clearData = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) {
      return;
    }
    setIsLoading(true);
    sendFile();
  }, [preview, sendFile]);

  const onDrop = (acceptedFiles) => {
    if (!acceptedFiles || acceptedFiles.length === 0) {
      setSelectedFile(undefined);
      setImage(false);
      setData(undefined);
      return;
    }
    setSelectedFile(acceptedFiles[0]);
    setData(undefined);
    setImage(true);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    maxFiles: 1
  });

  if (data) {
    confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  }

  return (
    <>
      <AppBar position="static" sx={{ background: "#2D890DFF", boxShadow: "none", color: "white" }}>
        <Toolbar>
          <Typography variant="h6" noWrap>
            AgroCure
          </Typography>
          <div style={{ flexGrow: 1 }} />
          <Avatar src={cblogo} />
        </Toolbar>
      </AppBar>
      <Container
        maxWidth={false}
        sx={{
          backgroundImage: `url(${bgImage})`,
          backgroundRepeat: "no-repeat",
          backgroundPosition: "center",
          backgroundSize: "cover",
          height: "93vh",
          marginTop: "8px",
        }}
        disableGutters
      >
        <Grid
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
          spacing={2}
          sx={{ padding: "4em 1em 0 1em" }}
        >
          <Grid item xs={12}>
            <Card
              sx={{
                margin: "auto",
                maxWidth: 400,
                height: image ? 550 : "auto",
                backgroundColor: "transparent",
                boxShadow: "0px 9px 70px 0px rgb(0 0 0 / 30%) !important",
                borderRadius: "15px",
              }}
            >
              {image && (
                <CardActionArea>
                  <CardMedia
                    component="img"
                    height="400"
                    image={preview}
                    alt="Uploaded plant leaf"
                  />
                </CardActionArea>
              )}
              {!image && (
                <CardContent>
                  <div
                    {...getRootProps()}
                    style={{
                      border: "2px dashed #cccccc",
                      borderRadius: "4px",
                      padding: "20px",
                      textAlign: "center",
                      cursor: "pointer",
                      backgroundColor: "rgba(255, 255, 255, 0.8)",
                    }}
                  >
                    <input {...getInputProps()} />
                    {isDragActive ? (
                      <p>Drop the image here ...</p>
                    ) : (
                      <p>Drag and drop an image of a potato plant leaf to process</p>
                    )}
                  </div>
                </CardContent>
              )}
              {data && (
                <CardContent
                  sx={{
                    backgroundColor: "white",
                    display: "flex",
                    justifyContent: "center",
                    flexDirection: "column",
                    alignItems: "center",
                  }}
                >
                  <TableContainer
                    component={Paper}
                    sx={{
                      backgroundColor: "transparent !important",
                      boxShadow: "none !important",
                    }}
                  >
                    <Table
                      sx={{
                        backgroundColor: "transparent !important",
                      }}
                    >
                      <TableHead
                        sx={{
                          backgroundColor: "transparent !important",
                        }}
                      >
                        <TableRow
                          sx={{
                            backgroundColor: "transparent !important",
                          }}
                        >
                          <TableCell
                            sx={{
                              fontSize: "22px",
                              backgroundColor: "transparent !important",
                              borderColor: "transparent !important",
                              color: "#000000a6 !important",
                              fontWeight: "bolder",
                              padding: "1px 24px 1px 16px",
                            }}
                          >
                            Disease
                          </TableCell>
                          <TableCell
                            sx={{
                              fontSize: "22px",
                              backgroundColor: "transparent !important",
                              borderColor: "transparent !important",
                              color: "#000000a6 !important",
                              fontWeight: "bolder",
                              padding: "1px 24px 1px 16px",
                            }}
                          >
                            Confidence
                          </TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody
                        sx={{
                          backgroundColor: "transparent !important",
                        }}
                      >
                        <TableRow
                          sx={{
                            backgroundColor: "transparent !important",
                          }}
                        >
                          <TableCell
                            sx={{
                              fontSize: "14px",
                              backgroundColor: "transparent !important",
                              borderColor: "transparent !important",
                              color: "#000000a6 !important",
                              fontWeight: "bolder",
                              padding: "1px 24px 1px 16px",
                            }}
                          >
                            {data.class}
                          </TableCell>
                          <TableCell
                            sx={{
                              fontSize: "14px",
                              backgroundColor: "transparent !important",
                              borderColor: "transparent !important",
                              color: "#000000a6 !important",
                              fontWeight: "bolder",
                              padding: "1px 24px 1px 16px",
                            }}
                          >
                            {confidence}%
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                  <ColorButton
                    variant="contained"
                    startIcon={<ClearIcon />}
                    onClick={clearData}
                    sx={{
                      width: "100%",
                      borderRadius: "15px",
                      padding: "15px 22px",
                      color: "#000000a6",
                      fontSize: "20px",
                      fontWeight: 900,
                    }}
                  >
                    Clear
                  </ColorButton>
                </CardContent>
              )}
              {isLoading && (
                <CardContent sx={{ display: "flex", justifyContent: "center" }}>
                  <CircularProgress sx={{ color: "#be6a77 !important" }} />
                </CardContent>
              )}
            </Card>
          </Grid>
        </Grid>
      </Container>
    </>
  );
};

function App() {
  return <ImageUpload />;
}

export default App;