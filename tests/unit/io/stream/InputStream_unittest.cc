/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <gtest/gtest.h>
#include <shogun/io/stream/FileInputStream.h>
#ifdef HAVE_LIBARCHIVE
#include <shogun/io/stream/ArchiveInputStream.h>
#endif

using namespace std;
using namespace shogun;

class InputStream : public ::testing::Test
{
public:
	io::FileSystemRegistry* fs_registry;
	virtual void SetUp()
	{
		fs_registry = io::FileSystemRegistry::instance();
	}

	virtual void TearDown() {}
};

TEST_F(InputStream, raw_file)
{
	unique_ptr<io::RandomAccessFile> file;
	std::string from("shogun-unit-test_test.cmake");
	auto r = fs_registry->new_random_access_file(from, &file);
	ASSERT_FALSE(r);
	auto fis = some<io::CFileInputStream>(file.get());
	string buffer;
	r = fis->read(&buffer, 10);
	ASSERT_FALSE(r);
	ASSERT_EQ("ADD_TEST (", buffer);
}

#ifdef HAVE_LIBARCHIVE
TEST_F(InputStream, archive_stream)
{
	unique_ptr<io::RandomAccessFile> file;
	std::string from("shogun-unit-test_test.cmake.gz");
	auto r = fs_registry->new_random_access_file(from, &file);
	ASSERT_FALSE(r);
	auto fis = some<io::CFileInputStream>(file.get());
	auto ais = some<io::CArchiveInputStream>(fis.get());
	string buffer;
	r = fis->read(&buffer, 10);
	cout << buffer << endl;
	ASSERT_FALSE(r);
}
#endif
