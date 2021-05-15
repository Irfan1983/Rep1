import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

public class readXml {

	public static void main(String[] args) throws SAXException, IOException, ParserConfigurationException 
	{
		File myFile = new File("C:\\Users\\Irfan\\eclipse-workspace\\ReadXml\\TestData\\123.csv"); 
		//File myFile = new File("C:\\Users\\Irfan\\Desktop\\123.csv"); 
		FileInputStream fis = new FileInputStream(myFile); // Finds the workbook instance for XLSX file
		HSSFWorkbook myWorkBook = new HSSFWorkbook (fis); // Return first sheet from the XLSX workbook 
		HSSFSheet mySheet = myWorkBook.getSheetAt(0);
		System.err.println(mySheet.getPhysicalNumberOfRows());
		for(int i =1;i<=mySheet.getPhysicalNumberOfRows()-1;i++)
		{
			
			String sno = mySheet.getRow(i).getCell(0).toString();
			String testscenario=mySheet.getRow(i).getCell(1).toString();
			String xmlpath=mySheet.getRow(i).getCell(2).toString();
			String keyword=mySheet.getRow(i).getCell(3).toString();
			String validation=mySheet.getRow(i).getCell(4).toString();
			//System.out.println(keyword);
			if(keyword.equalsIgnoreCase("yes"))
			{
				Object Path;
				Path p1 = Paths.get(mySheet.getRow(i).getCell(2).toString());
				String p2=p1.toString();
		        String[] arrOfStr = validation.split(",", 0);
		  
		        for (String a : arrOfStr)
		        {
					System.out.println(a+":"+verifyAllElement(p2,a));

		    }
			}
			
		}

		
		
	}
	
	public static boolean verifyAllElement(String FilePath,String validationData) throws ParserConfigurationException, SAXException, IOException
	{
		boolean isPresence = false;
		File file = new File(FilePath);  
		//an instance of factory that gives a document builder  
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();  
		//an instance of builder to parse the specified xml file  
		DocumentBuilder db = dbf.newDocumentBuilder();  
		Document doc = db.parse(file);  
		doc.getDocumentElement().normalize();  
		//System.out.println(doc.getAttributes());
		//System.out.println("Root element: " + doc.getDocumentElement().getNodeName());  
		NodeList list = doc.getElementsByTagName("*");
		
		//System.out.println("XML Elements: ");
	    for (int i=0; i<list.getLength(); i++) 
	    {
            Element element = (Element)list.item(i);
            if(element.getNodeName().contentEquals(validationData))
            {
            	 isPresence = true;
            	 //System.out.println(element.getNodeName());
            	 //System.out.println(validationData);
            	 //System.out.println(isPresence);
            }
          
  
	    }
	    return isPresence;
	}
	

}
